use eframe::egui;
use egui::Color32;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::process::Command as ProcessCommand;
use std::io::{self, Write};

use crate::model::{Model, ConversationMessage};
use crate::Config;
use crate::shell::Shell;

/// One message in the chat log.
#[derive(Serialize, Deserialize, Clone)]
pub struct ChatMessage {
    pub content: String,
    pub is_user: bool,
    pub timestamp: DateTime<Utc>,
    pub is_command: bool,
    pub executed: bool,
}

/// A full conversation with a unique id and title (first user prompt).
#[derive(Serialize, Deserialize, Clone)]
pub struct ChatSession {
    pub id: String,
    pub title: String,
    pub messages: Vec<ChatMessage>,
    pub created_at: DateTime<Utc>,
}

impl Default for ChatSession {
    fn default() -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            title: "New Chat".into(),
            messages: Vec::new(),
            created_at: Utc::now(),
        }
    }
}

/// Main GUI application state.
pub struct LlmTermApp {
    pub config: Config,
    pub current_input: String,
    pub current_session: ChatSession,
    pub chat_sessions: Vec<ChatSession>,
    pub selected_session_id: Option<String>,
    /// simple on‑disk cache; maps prompt -> response
    pub cache: HashMap<String, String>,
    pub is_loading: bool,
    pub pending_command: Option<String>,
}

impl LlmTermApp {
    /* --------------------------------------------------------------------- */
    /*                 ─── construction / persistence ───                    */
    /* --------------------------------------------------------------------- */

    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let chat_sessions = Self::load_chat_sessions().unwrap_or_default();

        Self {
            config: Config {
                model: Model::OpenAiGpt4oMini,
                max_tokens: 1000,
            },
            current_input: String::new(),
            current_session: ChatSession::default(),
            chat_sessions,
            selected_session_id: None,
            cache: HashMap::new(),
            is_loading: false,
            pending_command: None,
        }
    }

    fn sessions_file_path() -> std::io::Result<PathBuf> {
        let mut path = dirs::home_dir()
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::Other, "no home dir"))?;
        path.push(".llm_term_sessions.json");
        Ok(path)
    }

    fn load_chat_sessions() -> std::io::Result<Vec<ChatSession>> {
        let path = Self::sessions_file_path()?;
        if path.exists() {
            let content = fs::read_to_string(path)?;
            Ok(serde_json::from_str(&content).unwrap_or_default())
        } else {
            Ok(Vec::new())
        }
    }

    fn save_chat_sessions(&self) -> std::io::Result<()> {
        let path = Self::sessions_file_path()?;
        let content = serde_json::to_string_pretty(&self.chat_sessions)?;
        fs::write(path, content)
    }

    /* --------------------------------------------------------------------- */
    /*                          session management                           */
    /* --------------------------------------------------------------------- */

    fn new_chat(&mut self) {
        // push current session if it has any messages
        if !self.current_session.messages.is_empty() {
            self.chat_sessions.push(self.current_session.clone());
        }

        self.current_session = ChatSession::default();
        self.selected_session_id = None;
        self.current_input.clear();
    }

    fn load_session(&mut self, session_id: &str) {
        if let Some(pos) = self.chat_sessions.iter().position(|s| s.id == session_id) {
            // swap out
            let mut session = self.chat_sessions.remove(pos);
            std::mem::swap(&mut self.current_session, &mut session);
            self.chat_sessions.push(session); // save the replaced session back
            self.selected_session_id = Some(self.current_session.id.clone());
        }
    }

    /* --------------------------------------------------------------------- */
    /*               helper: execute a command                               */
    /* --------------------------------------------------------------------- */
    fn execute_command(&self, command: &str) -> String {
        let (shell_cmd, shell_arg) = Shell::detect().to_shell_command_and_command_arg();
        
        match ProcessCommand::new(shell_cmd)
            .arg(shell_arg)
            .arg(command)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
        {
            Ok(child) => {
                match child.wait_with_output() {
                    Ok(output) => {
                        let mut result = String::new();
                        if !output.stdout.is_empty() {
                            result.push_str(&String::from_utf8_lossy(&output.stdout));
                        }
                        if !output.stderr.is_empty() {
                            if !result.is_empty() {
                                result.push_str("\n");
                            }
                            result.push_str(&String::from_utf8_lossy(&output.stderr));
                        }
                        if result.is_empty() {
                            result = "Command executed successfully (no output)".to_string();
                        }
                        result
                    }
                    Err(e) => format!("Command execution failed: {}", e),
                }
            }
            Err(e) => format!("Failed to start command: {}", e),
        }
    }

    /* --------------------------------------------------------------------- */
    /*               helper: handle a user submitting a prompt               */
    /* --------------------------------------------------------------------- */
    fn handle_user_prompt(&mut self, ctx: &egui::Context, prompt: String) {
        // push user message
        self.current_session.messages.push(ChatMessage {
            content: prompt.clone(),
            is_user: true,
            timestamp: Utc::now(),
            is_command: false,
            executed: false,
        });

        if self.current_session.title == "New Chat" {
            self.current_session.title = prompt
                .chars()
                .take(30)
                .collect::<String>()
                .trim_end()
                .to_string();
        }

        // Check if this is a "yes" response to execute the last command
        if let Some(ref pending_cmd) = self.pending_command.clone() {
            let trimmed_prompt = prompt.trim().to_lowercase();
            if matches!(trimmed_prompt.as_str(), "yes" | "y" | "sure" | "go ahead" | "execute" | "run it" | "do it") {
                // Execute the pending command
                let output = self.execute_command(pending_cmd);
                self.pending_command = None;
                
                self.current_session.messages.push(ChatMessage {
                    content: format!("Executing: {}\n\nOutput:\n{}", pending_cmd, output),
                    is_user: false,
                    timestamp: Utc::now(),
                    is_command: false,
                    executed: false,
                });
                
                let _ = self.save_chat_sessions();
                ctx.request_repaint();
                return;
            }
        }

        // Convert current session messages to conversation history format
        let conversation_history: Vec<ConversationMessage> = self.current_session.messages
            .iter()
            .map(|msg| ConversationMessage {
                content: msg.content.clone(),
                is_user: msg.is_user,
            })
            .collect();

        let cache_key = format!("unified:{}", prompt);
        let response = match self.cache.get(&cache_key) {
            Some(cached) => cached.clone(),
            None => {
                match self.config.model.llm_get_response(&self.config, &prompt, true, &conversation_history) {
                    Ok(Some(reply)) => {
                        self.cache.insert(cache_key, reply.clone());
                        reply
                    }
                    Ok(None) => {
                        let reply = "I'm not sure how to respond to that.".to_string();
                        self.cache.insert(cache_key, reply.clone());
                        reply
                    }
                    Err(e) => {
                        let reply = format!("Error: {}", e);
                        self.cache.insert(cache_key, reply.clone());
                        reply
                    }
                }
            }
        };

        // Check if response contains a command or execute instruction
        if response.contains("EXECUTE_LAST_COMMAND") {
            if let Some(pending_cmd) = self.pending_command.clone() {
                let output = self.execute_command(&pending_cmd);
                self.pending_command = None;
                
                self.current_session.messages.push(ChatMessage {
                    content: format!("Executing: {}\n\nOutput:\n{}", pending_cmd, output),
                    is_user: false,
                    timestamp: Utc::now(),
                    is_command: false,
                    executed: false,
                });
            }
        } else {
            // Look for COMMAND: pattern in the response
            if let Some(cmd_start) = response.find("COMMAND: ") {
                let cmd_part = &response[cmd_start + 9..];
                if let Some(cmd_end) = cmd_part.find('`') {
                    let command = cmd_part[..cmd_end].trim();
                    self.pending_command = Some(command.to_string());
                }
            }
            
            self.current_session.messages.push(ChatMessage {
                content: response,
                is_user: false,
                timestamp: Utc::now(),
                is_command: false,
                executed: false,
            });
        }

        // persist
        let _ = self.save_chat_sessions();

        // keep UI responsive
        ctx.request_repaint();
    }
}

/* ------------------------------------------------------------------------- */
/*                             egui integration                              */
/* ------------------------------------------------------------------------- */

impl eframe::App for LlmTermApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        /* --------------- LEFT: history ---------------- */
        let mut new_chat_clicked = false;
        let mut session_to_load: Option<String> = None;

        egui::SidePanel::left("history").show(ctx, |ui| {
            ui.heading("Chat History");
            if ui.button("+ New Chat").clicked() {
                new_chat_clicked = true;
            }

            ui.separator();

            egui::ScrollArea::vertical().show(ui, |ui| {
                for session in &self.chat_sessions {
                    let selected = self
                        .selected_session_id
                        .as_ref()
                        .map(|id| id == &session.id)
                        .unwrap_or(false);

                    if ui.selectable_label(selected, &session.title).clicked() {
                        session_to_load = Some(session.id.clone());
                    }
                }
            });
        });

        if new_chat_clicked {
            self.new_chat();
        }
        if let Some(id) = session_to_load {
            self.load_session(&id);
        }

        /* --------------- CENTRAL: chat log ------------ */
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("LLM Terminal");
            ui.separator();

            egui::ScrollArea::vertical()
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    for msg in &self.current_session.messages {
                        ui.horizontal(|ui| {
                            if msg.is_user {
                                ui.colored_label(Color32::LIGHT_BLUE, "You:");
                            } else {
                                ui.colored_label(Color32::LIGHT_GREEN, "Assistant:");
                            }
                            ui.label(&msg.content);
                        });
                        ui.separator();
                    }
                    
                    // Show pending command indicator
                    if self.pending_command.is_some() {
                        ui.horizontal(|ui| {
                            ui.colored_label(Color32::YELLOW, "💬 Waiting for your response...");
                        });
                    }
                });
        });

        /* --------------- BOTTOM: composer ------------- */
        egui::TopBottomPanel::bottom("composer")
            .exact_height(40.0)
            .resizable(false)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label("💬");

                    let input =
                        ui.add(egui::TextEdit::singleline(&mut self.current_input).hint_text("Type here…"));

                    let send_pressed = ui
                        .add_enabled(!self.current_input.trim().is_empty(), egui::Button::new("Send"))
                        .clicked()
                        || (input.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)));

                    if send_pressed {
                        let prompt = std::mem::take(&mut self.current_input);
                        self.handle_user_prompt(ctx, prompt);
                    }
                });
            });
    }
}
