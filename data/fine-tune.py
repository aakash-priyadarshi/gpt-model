import openai

def fine_tune_model():
    # Set your OpenAI API key here
    openai.api_key = 'your-api-key'

    # Upload your training data
    try:
        training_file = openai.File.create(
            file=open("transformed_data.jsonl", "rb"),
            purpose='fine-tune'
        )
        training_file_id = training_file["id"]
        print(f"File uploaded successfully with ID: {training_file_id}")
    except Exception as e:
        print(f"Failed to upload file: {e}")
        return

    # Start the fine-tuning process
    try:
        fine_tune = openai.FineTune.create(
            model="gpt-3.5-turbo",
            training_file=training_file_id,
            n_epochs=4,
            batch_size=4,
            learning_rate_multiplier=0.1
        )
        fine_tune_id = fine_tune["id"]
        print(f"Fine-tuning started successfully with ID: {fine_tune_id}")

        # You can optionally poll for status updates
        status = openai.FineTune.retrieve(id=fine_tune_id)
        print(f"Fine-tuning status: {status['status']}")
    except Exception as e:
        print(f"Failed to start fine-tuning: {e}")

if __name__ == "__main__":
    fine_tune_model()
