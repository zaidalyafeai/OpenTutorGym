# AI Math Tutor

An interactive math tutoring application that simulates a conversation between a student and a tutor using AI models.

## Features

- Uses Ollama and OpenAI models for the student and tutor roles
- Interactive conversation with streaming responses
- Customizable problem statements and conversation length

## Debugging Features

This application includes several debugging features that allow you to inspect and troubleshoot the application without restarting it:

### Hot Reloading

The application uses Gradio's hot reloading feature, which automatically reloads the application when you make changes to the code. This allows you to make changes to the application without restarting it.

To use hot reloading:

1. Start the application with `python appv2.py`
2. Make changes to the code
3. The application will automatically reload with your changes

### Debug Panel

The application includes a debug panel that allows you to:

- Evaluate expressions in the application context
- View recent logs
- Check the status of available models

To use the debug panel:

1. Open the "Debug Panel" accordion in the UI
2. Enter an expression to evaluate (e.g., `available_models`, `student_model.value`)
3. Click "Run Debug" to see the result
4. Use "Refresh Logs" to view recent application logs
5. Use "Check Models" to see the status of available models

### Logging

The application includes comprehensive logging that captures:

- Function calls with parameters and return values
- Execution time for key operations
- Error information with stack traces

Logs are stored in the `logs` directory with timestamps.

### Remote Debugging

For advanced debugging, you can use the remote debugging feature:

1. Open your browser's developer console
2. Execute JavaScript to send debugging commands to the application:

```javascript
// Example: Check available models
fetch('/api/debug', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    data: [
      'result = get_available_models()'
    ]
  })
}).then(r => r.json()).then(console.log);
```

## Development

### Prerequisites

- Python 3.8+
- Ollama installed locally (for local models)
- OpenAI API key (for GPT models)

### Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Create a `.env` file with your API keys:
   ```
   OPENAI_API_KEY=your_openai_key
   DEEPSEEK_API_KEY=your_deepseek_key
   ```

### Running the Application

```bash
python appv2.py
```

The application will be available at http://localhost:7860 