# 2024 Voter Match Application

A Python-based web application that helps voters find political candidates who best align with their views on key policy issues.
Created by Carrie Andrews-Smith, Aysha Zayyad, Pawarisa Sears

## Overview

This application uses Natural Language Processing (NLP) to match voters with 2024 presidential candidates based on their stance on various political issues. The app employs a pre-trained Natural Language Inference (NLI) model to compare user responses with candidate positions and calculate alignment scores.

## Features

- Interactive interface for users to input their stance on key political issues
- Comprehensive policy topic coverage including:
  - Immigration
  - Healthcare
  - Energy and Environmental Issues
  - Economy
  - Education
  - Gun Regulation
  - Criminal Justice
  - Foreign Policy
  - Abortion
  - Other Policy Positions
- Real-time matching using NLP
- Detailed alignment scores for each candidate
- Links to additional candidate information

## Requirements

- Python 3.7+
- NLTK
- Gradio
- Transformers (Hugging Face)
- python-docx
- Other dependencies listed in requirements.txt

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd voter-match-app
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download required NLTK data:
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')
```

## Usage

1. Start the application:
```bash
python app_voter_match.py
```

2. Open your web browser and navigate to the local server address (typically http://localhost:7861)

3. Select your stance on each policy issue from the provided options

4. Click "Submit" to see your candidate matches

## Technical Details

### Components

- **NLP Processing**: Uses NLTK for text preprocessing including tokenization, lemmatization, and stop word removal
- **Matching Algorithm**: Employs RoBERTa-large-mnli model for Natural Language Inference
- **Web Interface**: Built with Gradio for an interactive user experience
- **Data Processing**: Handles candidate stance data from structured documents

### Data Structure

- Candidate data is organized by election year (2016, 2020, 2024)
- Each candidate has standardized policy positions across major topics
- User responses are matched against preprocessed candidate statements

### Scoring System

- Uses NLI model to generate alignment scores between user stances and candidate positions
- Calculates overall match percentages based on aggregated topic scores
- Provides detailed breakdown of alignment by policy area

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Your chosen license]

## Authors

[Your name/organization]

## Acknowledgments

- Data sources: Ballotpedia and official campaign websites
- Built using Hugging Face's Transformers library
- Gradio team for the web interface framework

## Note

This is an educational tool and should not be the sole basis for voting decisions. Users are encouraged to conduct their own research and verify candidate positions through official sources.
