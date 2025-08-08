from transformers import pipeline
#TODO: import the necessary libraries for NLI

def match_user_to_candidates(user_input, standardized_candidates):
    """
        Match user-chosen input with standardized candidates' stances using a text classification pipeline.

        Args:
        user_input (dict): User-chosen stances for each topic.
        standardized_candidates (dict): Standardized candidates' stances for each topic.

        Returns:
        dict: A dictionary containing the match results with labels and scores.
    """

    results = {}
    for candidate, topics in standardized_candidates.items():
        results[candidate] = {}
        for topic, statements in topics.items():
            if topic in user_input:
                user_stance = user_input[topic]
                combined_statements = " ".join(statements)  # Combine candidate stances
                if combined_statements:  # Avoid empty premise
                    result = nli_pipeline(
                        f"Premise: {combined_statements} Hypothesis: {user_stance}",
                        truncation=True
                    )[0]
                    label = result["label"]
                    score = result["score"]
                    results[candidate][topic] = {"label": label, "score": score}
    return results


def aggregate_scores(match_results):
    """
        Aggregate scores from match results to compute overall scores for each candidate.

        Args:
        match_results (dict): Match results containing labels and scores for each candidate and topic.

        Returns:
        list: A list of tuples containing candidates and their average scores, sorted by score.
    """
    
    overall_scores = []

    for candidate, topics in match_results.items():
        total_score = 0
        count = 0

        for result in topics.values():
            label = result["label"]
            score = result["score"]

            if label == "ENTAILMENT" and score > 0.4:
                total_score += 1
                count += 1
            elif label == "ENTAILMENT":
                total_score += score
                count += 1
            elif label == "CONTRADICTION" and score > 0.4:
                total_score -= (0.1*score)
                #count += 1
            elif label == "NEUTRAL" and score > 0.4:
                total_score += (0.1*score)
                count += 1
            else:
                total_score += 0
                count += 1

        average_score = total_score / count if count else 0
        overall_scores.append((candidate, average_score))

    overall_scores.sort(key=lambda x: x[1], reverse=True)
    return overall_scores