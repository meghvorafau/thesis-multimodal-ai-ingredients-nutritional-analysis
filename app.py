"""
Streamlit application for detecting ingredients in a food image and estimating
their lactose, fructose and sorbitol content.  The app wraps the processing
pipeline defined in ``ingredient_nutrient_analysis.py`` and provides an
interactive interface for uploading an image, running the detection and
viewing the results.

To run this app locally, install the required dependencies (streamlit and
pillow) and launch it with ``streamlit run streamlit_app.py``.  The models
used for detection and embedding are loaded once and cached for subsequent
requests to avoid repeated heavy initialisation.  Similarly, the BLS
database and its embeddings are cached.
"""

from __future__ import annotations

import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import torch
from typing import List, Dict, Tuple

from ingredient_nutrient_analysis import (
    ModelLoader,
    BERTEmbedder,
    extract_ingredients_with_quantities,
    parse_quantity,
    fuzzy_match_ingredient,
    get_nutrient_columns,
    estimate_nutrients,
    normalize_embeddings,
    free_memory,
    clean_ingredient_name,
    DEFAULT_WEIGHT,
    CONFIG,
)

# ---------------------------------------------------------------------------
# Adjust configuration for streamlit environment
#
# The BLS path may differ from the default in the analysis module.  We update
# it here so that the underlying pipeline uses the correct location of the
# uploaded BLS database.  By default, this file is stored in the /home/oai/share
# directory for the streamlit runner.
CONFIG["bls_path"] = "/home/oai/share/BLS_3.02_BesserEsser_englisch.csv"


@st.cache_resource(show_spinner=False)
def load_models_cached() -> Tuple[
    object,  # LlavaProcessor
    object,  # LlavaForConditionalGeneration
    BERTEmbedder,
]:
    """Load and cache the LLaVA and BERT models for reuse across sessions."""
    processor, llava_model, bert_model, tokenizer = ModelLoader.load_models()
    embedder = BERTEmbedder(bert_model, tokenizer)
    return processor, llava_model, embedder


@st.cache_resource(show_spinner=False)
def load_bls_cached(embedder: BERTEmbedder) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, str]]:
    """
    Load the BLS database, compute or load its embeddings and determine the
    nutrient columns for lactose, fructose and sorbitol.  The embeddings and
    normalised embeddings are computed once per session.  Returns the
    dataframe, the matrix of normalised embeddings and a mapping of nutrient
    names to column names.
    """
    bls_df = pd.read_csv(CONFIG["bls_path"], encoding="latin1", usecols=None)
    bls_df["english_name"] = bls_df.get("Textenglisch", bls_df.get("displayname", "")).fillna(
        bls_df.get("displayname", "")
    )
    bls_df["german_name"] = bls_df.get("displayname", bls_df.get("Textenglisch", "")).fillna(
        bls_df.get("Textenglisch", "")
    )
    bls_df["clean_english"] = bls_df["english_name"].apply(clean_ingredient_name)
    # Compute embeddings for BLS entries
    bls_embeddings = embedder.embed(bls_df["clean_english"].tolist()).numpy()
    normalized_embeddings = normalize_embeddings(bls_embeddings)
    nutrient_cols = get_nutrient_columns(bls_df)
    return bls_df, normalized_embeddings, nutrient_cols


def run_pipeline(image: Image.Image) -> Dict[str, object]:
    """
    Execute the full pipeline on a given image.  Returns a dictionary
    containing ingredient matches, estimated weights, nutrient contributions
    per ingredient and total nutrient sums.
    """
    results: Dict[str, object] = {
        "ingredients": [],
        "totals": {"lactose": 0.0, "fructose": 0.0, "sorbitol": 0.0},
    }
    # Load models
    processor, llava_model, embedder = load_models_cached()
    # Prepare prompt and inputs
    prompt = "<image>\nList visible food ingredients as comma-separated names:\nIngredients:"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(
        CONFIG["device"], CONFIG["dtype"]
    )
    # Generate ingredients list
    with torch.no_grad():
        output = llava_model.generate(
            **inputs,
            max_new_tokens=CONFIG["max_new_tokens"],
            temperature=0.4,
            do_sample=True,
            pad_token_id=processor.tokenizer.eos_token_id,
            num_beams=2,
            early_stopping=True,
        )
    response = processor.decode(output[0], skip_special_tokens=True)
    ingredients_with_orig = extract_ingredients_with_quantities(response)
    # Free memory from LLaVA
    free_memory()
    # Load BLS data and embeddings
    bls_df, normalized_embeddings, nutrient_cols = load_bls_cached(embedder)
    bls_names = bls_df["clean_english"].tolist()
    # Embed detected ingredients and compute similarity
    if ingredients_with_orig:
        clean_names = [ing for ing, _orig in ingredients_with_orig]
        ingredient_embeddings = embedder.embed(clean_names).numpy()
        ingredient_embeddings_norm = normalize_embeddings(ingredient_embeddings)
        similarities = ingredient_embeddings_norm.dot(normalized_embeddings.T)
    else:
        similarities = np.zeros((0, len(bls_df)))
    # Process each ingredient
    for (ing_clean, orig), scores in zip(ingredients_with_orig, similarities):
        # Determine top semantic matches
        top_indices = np.argsort(scores)[-CONFIG["top_k_matches"] :][::-1]
        top_scores = scores[top_indices]
        top_matches = bls_df.iloc[top_indices]
        # Determine confidence and possible fuzzy match
        is_high_confidence = top_scores[0] >= CONFIG["similarity_threshold"]
        confidence = "✓ HIGH CONFIDENCE" if is_high_confidence else "✗ LOW CONFIDENCE"
        fuzzy_match_res = None
        fuzzy_row = None
        if not is_high_confidence:
            result = fuzzy_match_ingredient(ing_clean, bls_names)
            if result:
                fuzzy_match_res, fuzzy_score, fuzzy_idx = result
                if fuzzy_score > CONFIG["fuzzy_threshold"]:
                    confidence = "⚡ FUZZY MATCH"
                    is_high_confidence = True
                    fuzzy_row = bls_df.iloc[fuzzy_idx]
        # Record matches
        match_list: List[Dict[str, object]] = []
        for rank, (score, match) in enumerate(zip(top_scores, top_matches.itertuples()), start=1):
            match_list.append(
                {
                    "rank": rank,
                    "english": match.english_name,
                    "german": match.german_name,
                    "id": match.INGREDIENT_ID,
                    "score": float(score),
                }
            )
        # Determine weight
        weight = parse_quantity(orig)
        if weight is None:
            weight = DEFAULT_WEIGHT
        # Compute nutrient contributions
        nutrient_estimates = estimate_nutrients(match_list, bls_df, nutrient_cols, weight)
        for nutrient, amount in nutrient_estimates.items():
            results["totals"][nutrient] += amount
        # Build result entry
        results["ingredients"].append(
            {
                "name": ing_clean,
                "original": orig,
                "confidence": confidence,
                "matches": match_list,
                "weight_g": weight,
                "nutrients": nutrient_estimates,
            }
        )
    return results


def display_results(results: Dict[str, object]) -> None:
    """Render the pipeline results in the streamlit UI."""
    if not results["ingredients"]:
        st.warning("No ingredients detected.")
        return
    # Show table with ingredients, weight and nutrient contributions
    table_data = []
    for item in results["ingredients"]:
        row = {
            "Ingredient": item["name"],
            "Estimated Weight (g)": f"{item['weight_g']:.1f}",
        }
        for nutrient, amount in item["nutrients"].items():
            row[f"{nutrient.capitalize()} (g)"] = f"{amount:.2f}"
        table_data.append(row)
    st.subheader("Detected Ingredients and Estimated Nutrients")
    st.table(table_data)
    # Show totals
    st.subheader("Total Estimated Nutrients in Dish")
    for nutrient, total in results["totals"].items():
        st.metric(label=nutrient.capitalize(), value=f"{total:.2f} g")


def main() -> None:
    """Main function for the streamlit app."""
    st.title("Food Ingredient & Nutrient Estimator")
    st.write(
        "Upload a photo of a prepared dish and the app will attempt to "
        "identify the visible ingredients, estimate their weight and compute "
        "an approximate lactose, fructose and sorbitol content based on the "
        "Bundeslebensmittelschlüssel (BLS) database."
    )
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded image", use_column_width=True)
        if st.button("Analyse"):
            with st.spinner("Running analysis..."):
                results = run_pipeline(image)
            display_results(results)


if __name__ == "__main__":
    main()
