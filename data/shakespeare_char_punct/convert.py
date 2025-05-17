import string
import argparse

def transform_file(filename):
  """
  Transforms a file in-place, applying the following transformations:
    - Punctuation becomes '1'
    - Vowels become '2'
    - Consonants become '3'
    - Everything else becomes '_'

  Args:
    filename: The name of the file to transform.
  """

  punctuation = string.punctuation
  vowels = 'aeiouAEIOU'
  consonants = ''.join([c for c in string.ascii_letters if c not in vowels])

  try:
    with open(filename, 'r+') as file:
      # Read the entire file content
      file_content = file.read()

      transformed_content = ''
      for char in file_content:
        if char in punctuation:
          transformed_content += '1'
        elif char in vowels:
          transformed_content += '2'
        elif char in consonants:
          transformed_content += '3'
        else:
          transformed_content += '_'

      # Go back to the beginning of the file
      file.seek(0)
      # Write the transformed content
      file.write(transformed_content)
      # Truncate the file in case the new content is shorter
      file.truncate()

  except FileNotFoundError:
    print(f"Error: File '{filename}' not found.")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Transform a text file by replacing characters.")
  parser.add_argument("input_file", help="The input text file to transform.")
  args = parser.parse_args()

  transform_file(args.input_file)
