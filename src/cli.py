from assessment.analysis import analyse_answers
from content_generation.edu_content_perplexity import generate_text, generate_questions


def main():
    topic = input("Topic: ")

    text = generate_text(topic)
    questions = generate_questions(text)

    print("Read the text and answer the questions.\n\n")
    print(text)
    print("\n\n")
    for index, question in enumerate(questions):
        print(f"{index+1}. {question}")

    answers = {}  # {0: (handwritten_answer_filepath, spoken_answer_filepath), 1: ...}

    while len(answers.keys()) < len(questions):
        question_number = input("Select a question number: ")

        try :
            question_number = int(question_number)
        except ValueError:
            print("Please enter a valid question number.")

        if question_number > len(questions):
            print("Invalid question number. Please select a valid one.")
            continue

        if question_number in answers.keys():
            print("You have already answered this question. Please select another one.")
            continue

        question = questions[int(question_number)-1]

        answer_handwritten_filepath = input(f"Question {question_number}: Write the answer and provide a filepath:\n")
        answer_spoken_filepath = input(f"Question {question_number}: Record the answer and provide a filepath:\n")
        answers[int(question_number)-1] = (answer_handwritten_filepath, answer_spoken_filepath)

        feedback = analyse_answers(
            text,
            question,
            handwritten_answer_filepath=answer_handwritten_filepath,
            spoken_answer_filepath=answer_spoken_filepath,
        )

        print(feedback)

if __name__ == "__main__":
    main()
