import time
import gradio as gr
from utility.scoring import match_user_to_candidates, aggregate_scores


def gradio_interface(*args):
    start_time = time.time()
    user_input = {topic: args[i] for i, topic in enumerate(choices.keys())}
    match_results = match_user_to_candidates(user_input, standardized_candidates)
    overall_scores = aggregate_scores(match_results)

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Find the best matched candidate
    best_matched_candidate = overall_scores[0][0]
    best_matched_candidate_url = candidate_urls.get(best_matched_candidate, "#")

    formatted_results = "## Policy Scores of Matched Candidate\n"
    for topic, result in match_results[best_matched_candidate].items():
        formatted_results += f"- **{topic}**: {result['label']} (Score: {result['score']:.2f})\n"
    formatted_results += "\n"

    formatted_results += "### Your Overall Candidate Alignment\n"
    for candidate, score in overall_scores:
        formatted_results += f"- **{candidate}**: {score:.2f}\n"

    formatted_results += f"\n**Processing Time**: {elapsed_time:.2f} seconds\n"
    formatted_results += f"\n**To learn more about your best match Click on this URL**: [Visit {best_matched_candidate}]({best_matched_candidate_url})"

    return formatted_results



inputs = []
for topic, options in choices.items():
    inputs.append(gr.Radio(choices= options, label = f"What's your stance on {topic}?"))

outputs = gr.Markdown(label = "Match Results")

iface = gr.Interface(fn = gradio_interface, 
                     inputs = inputs, 
                     outputs = outputs, 
                     title = "🗳️ Voter's Match App ", 
                     description="Welcome to the 2024 Political Candidate Match App. Find out which candidate aligns most with your views. Please select your stance on each topic and click 'Submit' to see the results.")

iface.launch(server_port= 7861, share= True, debug= True)