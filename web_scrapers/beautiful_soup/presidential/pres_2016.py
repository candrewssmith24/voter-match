import requests
import time
from bs4 import BeautifulSoup
from docx import Document
from requests.exceptions import RequestException

# Set a custom User-Agent to mimic a browser request
HEADERS = {
    "User-Agent": "Chrome/58.0.3029.110"
}

# Phrases to exclude from the beginning of stance text
EXCLUDE_START_PHRASES = [
    "Email *", "First Name", "This page was current", "Ballotpedia's scope changes",
    "Please complete the Captcha", "Follow Ballotpedia", "Share this page"
]

# Phrases to exclude from the end of stance text
EXCLUDE_END_PHRASES = [
    "The link below is to the most recent stories", "Ballotpedia features",
    "Click here to contact", "For media inquiries", "please donate",
    "Post-debate analysis", "Communications:", "External Relations:", "Operations:",
    "Policy:", "Content Strategy:", "Tech:"
]

def get_candidate_names(candidates_url):
    response = requests.get(candidates_url, headers=HEADERS, timeout=10)
    soup = BeautifulSoup(response.text, 'html.parser')

    candidate_links = soup.find_all(
        'a', href=lambda href: href and "_presidential_campaign,_2016" in href and
        "possible" not in href and "vice_presidential_campaign" not in href
    )
    candidates = {}
    seen_urls = set()

    for link in candidate_links:
        candidate_name = link.get_text(strip=True)
        candidate_url = link['href']

        # Ensure the URL is complete
        if not candidate_url.startswith('http'):
            candidate_url = f"https://ballotpedia.org{candidate_url}"

        # Replace all spaces with underscores in the candidate URL
        candidate_url = candidate_url.replace(" ", "_")

        # Exclude vice presidential campaign URLs explicitly
        if "vice_presidential_campaign" in candidate_url:
            continue

        if candidate_url in seen_urls or not candidate_name:
            continue

        candidates[candidate_name] = candidate_url
        seen_urls.add(candidate_url)

    return candidates


def get_all_topic_links(candidate_url):
    try:
        response = requests.get(candidate_url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')

        topic_links = {}
        print(f"Extracting all topic links from {candidate_url}...")

        for link in soup.find_all('a', href=True):
            topic_url = link['href']
            topic_name = link.get_text(strip=True)

            # Ensure the topic URL is complete and replace spaces with underscores
            if "_presidential_campaign,_2016/" in topic_url:
                if not topic_url.startswith('http'):
                    topic_url = f"https://ballotpedia.org{topic_url}"

                # Replace all spaces with underscores in the topic URL
                topic_url = topic_url.replace(" ", "_")

                # Exclude vice presidential campaign URLs explicitly
                if "vice_presidential_campaign" in topic_url:
                    continue

                if topic_name and topic_name not in topic_links:
                    topic_links[topic_name] = topic_url
                    print(f"Found topic: {topic_name} - URL: {topic_url}")

        return topic_links
    except requests.exceptions.RequestException as e:
        print(f"Error accessing {candidate_url}: {e}")
        return {}


def clean_text(text):
    # Exclude paragraphs containing unwanted start or end phrases
    for phrase in EXCLUDE_START_PHRASES:
        if text.lower().startswith(phrase.lower()):
            return None
    for phrase in EXCLUDE_END_PHRASES:
        if phrase.lower() in text.lower():
            return None
    return text

def get_topic_stance(topic_url):
    try:
        response = requests.get(topic_url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove unwanted elements like headers, footers, navigation, and scripts
        for element in soup(['header', 'footer', 'aside', 'nav', 'script']):
            element.decompose()

        # Remove all citation links with href starting with "#cite_note"
        for cite_link in soup.find_all('a', href=lambda href: href and href.startswith('#cite_note')):
            cite_link.decompose()

        # Find all <p> elements (ignore <li> elements)
        paragraphs = soup.find_all('p')
        stance_text = []
        scraping = False

        for element in paragraphs:
            text = element.get_text(strip=True)

            # Skip paragraphs that start with unwanted phrases
            if any(start_phrase in text for start_phrase in EXCLUDE_START_PHRASES):
                scraping = True
                continue

            # Stop scraping when an unwanted end phrase is encountered
            if any(end_phrase in text for end_phrase in EXCLUDE_END_PHRASES):
                scraping = False
                break

            # Collect text if we are in scraping mode
            if scraping:
                cleaned_text = clean_text(text)
                if cleaned_text:
                    stance_text.append(cleaned_text)

        # Join the cleaned text into a single string
        stance = ' '.join(stance_text).strip()

        # Check if any stance text was extracted
        if not stance:
            print(f"No stance found for topic at {topic_url}")
            return None

        return stance
    except requests.exceptions.RequestException as e:
        print(f"Error accessing topic URL {topic_url}: {e}")
        return None

def write_to_word(candidates_stances):
    doc = Document()

    for candidate, topics in candidates_stances.items():
        doc.add_heading(candidate, level=1)
        for topic, stance in topics.items():
            doc.add_heading(topic, level=2)
            doc.add_paragraph(stance)

    doc.save('Presidential_Candidate_Stances_2016.docx')
    print("\nPolicy stances have been written to 'Presidential_Candidate_Stances_2016.docx'.")

def main():
    base_url = 'https://ballotpedia.org/Presidential_election,_2016'
    try:
        response = requests.get(base_url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
    except requests.exceptions.RequestException as e:
        print(f"Error accessing base URL: {e}")
        return

    candidates_link = soup.find('a', href=lambda href: href and "/Presidential_candidates,_2016" in href)
    if not candidates_link:
        print("No link to 'Presidential candidates, 2016' page found.")
        return

    candidates_url = f"https://ballotpedia.org{candidates_link['href']}"

    candidates = get_candidate_names(candidates_url)
    if not candidates:
        print("No candidates were found.")
        return

    all_stances = {}
    for name, url in candidates.items():
        topic_links = get_all_topic_links(url)
        candidate_stances = {}

        for topic, topic_url in topic_links.items():
            stance = get_topic_stance(topic_url)
            if stance:
                candidate_stances[topic] = stance
            else:
                print(f"No stance found for topic {topic} at {topic_url}")

        if candidate_stances:
            all_stances[name] = candidate_stances

        time.sleep(1)

    write_to_word(all_stances)

    print("\nExtracted Stances for Each Candidate:")
    for candidate, topics in all_stances.items():
        print(f"\nCandidate: {candidate}")
        for topic, stance in topics.items():
            print(f"  Topic: {topic}")
            print(f"    Stance: {stance}")

if __name__ == '__main__':
    main()