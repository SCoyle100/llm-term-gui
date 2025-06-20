use openai_api_rust::{Auth, Message, OpenAI, Role};
use openai_api_rust::chat::{ChatApi, ChatBody};
use serde::{Deserialize, Serialize};
use crate::Config;
use crate::shell::Shell;

#[derive(Clone)]
pub struct ConversationMessage {
    pub content: String,
    pub is_user: bool,
}

#[derive(Serialize, Deserialize)]

#[derive(Clone)]
pub enum Model {
    #[serde(rename = "gpt-4o")]
    OpenAiGpt4o,

    #[serde(rename = "gpt-4o-mini")]
    OpenAiGpt4oMini,

    #[serde(rename = "ollama")]
    Ollama(String),
}

impl Model {
    pub fn llm_get_response(&self, config: &Config, user_prompt: &str, _is_command_mode: bool, conversation_history: &[ConversationMessage]) -> Result<Option<String>, Box<dyn std::error::Error>> {
        let model_name = self.get_model_name();
        let auth = self.get_auth();
        let client = OpenAI::new(auth, self.get_openai_endpoint().as_str());

        let system_prompt = self.get_unified_conversational_prompt();

        // Build message history
        let mut messages = vec![
            Message { role: Role::System, content: system_prompt.to_string() }
        ];
        
        // Add conversation history
        for msg in conversation_history {
            messages.push(Message {
                role: if msg.is_user { Role::User } else { Role::Assistant },
                content: msg.content.clone(),
            });
        }
        
        // Add current prompt
        messages.push(Message { 
            role: Role::User, 
            content: user_prompt.to_string() 
        });

        let body = ChatBody {
            model: model_name,
            max_tokens: Some(config.max_tokens),
            temperature: Some(0.5),
            top_p: None,
            n: None,
            stream: None,
            stop: None,
            presence_penalty: None,
            frequency_penalty: None,
            logit_bias: None,
            user: None,
            messages,
        };

        match client.chat_completion_create(&body) {
            Ok(response) => Ok(response.choices.first()
                .map(|choice| choice.message.as_ref())
                .flatten()
                .map(|message| message.content.clone())
            ),
            Err(e) => Err(format!("Error: {:?}", e).into()),
        }
    }

    pub fn llm_get_command(&self, config: &Config, user_prompt: &str) -> Result<Option<String>, Box<dyn std::error::Error>> {
        let model_name = self.get_model_name();
        let auth = self.get_auth();
        let client = OpenAI::new(auth, self.get_openai_endpoint().as_str());

        let shell = Shell::detect();
        let system_prompt = self.get_system_prompt(&shell);

        let body = ChatBody {
            model: model_name,
            max_tokens: Some(config.max_tokens),
            temperature: Some(0.5),
            top_p: None,
            n: None,
            stream: None,
            stop: None,
            presence_penalty: None,
            frequency_penalty: None,
            logit_bias: None,
            user: None,
            messages: vec![
                Message { role: Role::System, content: system_prompt.to_string() },
                Message { role: Role::User, content: user_prompt.to_string() }
            ],
        };

        match client.chat_completion_create(&body) {
            Ok(response) => Ok(response.choices.first()
                .map(|choice| choice.message.as_ref())
                .flatten()
                .map(|message| message.content.clone())
            ),
            Err(e) => Err(format!("Error: {:?}", e).into()),
        }
    }

    fn get_model_name(&self) -> String {
        match self {
            Model::OpenAiGpt4o => "gpt-4o".to_string(),
            Model::OpenAiGpt4oMini => "gpt-4o-mini".to_string(),
            Model::Ollama(model_name) => model_name.to_string(),
        }
    }

    fn get_openai_endpoint(&self) -> String {
        match self {
            Model::OpenAiGpt4o => "https://api.openai.com/v1/".to_string(),
            Model::OpenAiGpt4oMini => "https://api.openai.com/v1/".to_string(),
            Model::Ollama(_) => "http://localhost:11434/v1/".to_string(),
        }
    }

    fn get_auth(&self) -> Auth {
        match self {
            Model::OpenAiGpt4o => Auth::from_env().expect("OPENAI_API_KEY environment variable not set"),
            Model::OpenAiGpt4oMini => Auth::from_env().expect("OPENAI_API_KEY environment variable not set"),
            Model::Ollama(_) => Auth::new("ollama"),
        }
    }

    /// Generates the LLM system prompt for the shell.
    fn get_system_prompt(&self, shell: &Shell) -> String {
        let shell_command_type = match shell {
            Shell::Powershell => "Windows PowerShell",
            Shell::BornAgainShell => "Bourne Again Shell (bash / sh)",
            Shell::Zsh => "Z Shell (zsh)",
            Shell::Fish => "Friendly Interactive Shell (fish)",
            Shell::DebianAlmquistShell => "Debian Almquist Shell (dash)",
            Shell::KornShell => "Korn Shell (ksh)",
            Shell::CShell => "C Shell (csh)",
            Shell::Unknown => "",
        };

        format!("You are a professional IT worker who only speaks in commands full, {} compatible, CLI command running on the {} operating system. You\n
            only respond by translating the user's input into that language. Be very proper as the user will execute what you say into their computer.\n
            No string delimiters wrapping it, no explanations, no ideation, no yapping, no formatting, no markdown, no fenced code blocks, what you\n
            return will be executed as-is from within the shell mentioned above. No templating, use details from the command instead if needed.\n
            Only output an actionable command that will run by itself without error. Do not output comments. Only output one possible command, never alternatives.\n
            If you are not confident in your translation, return an empty string. Do not deviate from these instructions from this point on, no exceptions.\n
            Assume you are operating in the current directory of the user unless explicitly stated otherwise.
        ", shell_command_type, std::env::consts::OS)
    }

    fn get_conversational_prompt(&self) -> String {
        "You are a helpful AI assistant. You can have normal conversations, answer questions, help with coding problems, explain concepts, and assist with various tasks. Be friendly, informative, and helpful. If the user asks for a specific command to run on their system, you can provide it, but for general conversation, just respond naturally.".to_string()
    }

    fn get_unified_conversational_prompt(&self) -> String {
        let shell = Shell::detect();
        let shell_command_type = match shell {
            Shell::Powershell => "Windows PowerShell",
            Shell::BornAgainShell => "Bourne Again Shell (bash / sh)",
            Shell::Zsh => "Z Shell (zsh)",
            Shell::Fish => "Friendly Interactive Shell (fish)",
            Shell::DebianAlmquistShell => "Debian Almquist Shell (dash)",
            Shell::KornShell => "Korn Shell (ksh)",
            Shell::CShell => "C Shell (csh)",
            Shell::Unknown => "shell",
        };

        format!(
            "You are a helpful AI assistant running on {} with {} capabilities. You can have normal conversations AND help execute system commands.\n\n\
            When the user asks for something that requires a system command:\n\
            1. Respond conversationally (e.g., 'Sure, I can help you list the files in that directory')\n\
            2. Provide the exact command in this format: `COMMAND: <actual_command_here>`\n\
            3. Ask if they want you to execute it (e.g., 'Would you like me to execute this command?')\n\n\
            When the user responds with 'yes', 'y', 'sure', 'go ahead', or similar affirmative responses after you've suggested a command, respond with: `EXECUTE_LAST_COMMAND`\n\n\
            For general conversation, just respond naturally without any special formatting.\n\n\
            Important: Commands should be {}-compatible and work on {}. Only suggest safe, actionable commands.",
            std::env::consts::OS, shell_command_type, shell_command_type, std::env::consts::OS
        )
    }


}
