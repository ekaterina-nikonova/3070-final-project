from assessment.analysis_local import analyse_answers
from content_generation.edu_content_perplexity import generate_text, generate_questions


def main():
    """Command-line interface prototype.
    
    This CLI prototype verifies the correctness of the software components, but due to
    the slow workflow involving manual uploading of handwritten answer's image and
    spoken audio file it is intended to be replaced by a GUI in production.
    """

    # Prompt the user to enter a topic for generating educational content
    topic = input("Topic: ")

    # Generate the text based on the provided topic using Perplexity API
    # and questions based on the generated text
    text = generate_text(topic)
    questions = generate_questions(text)

    # Display instructions and the generated text to the user
    print("Read the text and answer the questions.\n\n")
    print(text)
    print("\n\n")
    
    # Display each question with its number (1-indexed for display purposes only!)
    for index, question in enumerate(questions):
        print(f"{index+1}. {question}")

    # Initialize dictionary to store answers: key=question index, value=tuple of file paths with answers
    answers = {}  # {0: (handwritten_answer_filepath, spoken_answer_filepath), 1: ...}

    # Continue collecting answers until all questions have been answered
    while len(answers.keys()) < len(questions):
        # Prompt user to select which question they want to answer
        question_number = input("Select a question number: ")

        # Attempt to convert the input to an integer
        try :
            question_number = int(question_number)
        # Handle non-numeric input gracefully (the user can try again without crashing the program)
        except ValueError:
            print("Please enter a valid question number.")

        # Validate that the question number is within the valid range
        if question_number > len(questions):
            print("Invalid question number. Please select a valid one.")
            # Skip to next iteration to re-prompt for a valid question number
            continue

        # Check if this question has already been answered to prevent duplicates
        if question_number in answers.keys():
            print("You have already answered this question. Please select another one.")
            # Skip to next iteration to prompt for a different question
            continue

        # Retrieve the question text converting back to the 0-based index
        question = questions[int(question_number)-1]

        # Prompt user to provide file path to the handwritten answer image and the spoken answer audio file
        answer_handwritten_filepath = input(f"Question {question_number}: Write the answer and provide a filepath:\n")
        answer_spoken_filepath = input(f"Question {question_number}: Record the answer and provide a filepath:\n")
        # Store both file paths in the answers dictionary with 0-based question index
        answers[int(question_number)-1] = (answer_handwritten_filepath, answer_spoken_filepath)

        # Analyze the provided answers by processing OCR on handwritten text and ASR on audio.
        # Compare results against the generated text and question to generate feedback
        feedback = analyse_answers(
            text,                                                     # The original educational text for context
            question,                                                 # The specific question being answered
            handwritten_answer_filepath=answer_handwritten_filepath,  # Path to handwritten image
            spoken_answer_filepath=answer_spoken_filepath,            # Path to audio recording
        )

        # Display the analysis feedback to the user
        print(feedback)


if __name__ == "__main__":
    # Execute the main function when script is run directly
    main()
