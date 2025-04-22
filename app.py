import streamlit as st
from huggingface_hub import InferenceClient
import re



# Replace with your actual Hugging Face token (or use st.secrets["HF_TOKEN"] for security)
HF_TOKEN = "Replace with your actual Hugging Face token"

# Initialize the inference client
client = InferenceClient(
    model="flax-community/t5-recipe-generation",
    token=HF_TOKEN
)

def generate_recipes(ingredients_list):
    input_text = "ingredients: " + ', '.join(ingredients_list)

    responses = []
    # Generate 5 recipes using sampling
    for _ in range(5):
        try:
            output = client.text_generation(
                prompt=input_text,
                max_new_tokens=200,
                temperature=0.9,
                top_k=50,
                do_sample=True
            )
            responses.append(output.strip())
        except Exception as e:
            st.error(f"API call failed: {e}")
            break  # Stop further attempts if API call fails

    return responses



def parse_recipe(recipe_text):
    # Try to extract title, ingredients, and directions from the recipe text
    title = "No title found."
    ingredients = "No ingredients listed."
    directions = "No directions listed."

    # Extract the title after 'title:'
    title_match = re.search(r'title:\s*(.*?)(?=ingredients:|$)', recipe_text, re.IGNORECASE | re.DOTALL)
    if title_match:
        title = title_match.group(1).strip()

    # Extract ingredients after 'ingredients:' and before 'directions:'
    ingredients_match = re.search(r'ingredients?: ingredients\s*(.*?)(?= directions:|$)', recipe_text, re.IGNORECASE | re.DOTALL)
    if ingredients_match:
        ingredients = ingredients_match.group(1).strip()

    # Extract directions after 'directions:'
    directions_match = re.search(r'directions?:\s*(.*)', recipe_text, re.IGNORECASE | re.DOTALL)
    if directions_match:
        directions = directions_match.group(1).strip()

    return title, ingredients, directions


# Streamlit UI
st.set_page_config(page_title="Smart Recipe Generator", page_icon="🍳")
st.title("👩‍🍳 Smart Recipe Generator 🍽️")
st.markdown("Create delicious recipes from your selected ingredients using AI!")

st.header("🧂 Choose Your Ingredients")
ingredients = [
    'Tomato', 'Onion', 'Milk', 'Ghee', 'Chicken', 'Rice', 
    'Potato', 'Butter', 'Coriander', 'Garlic', 'Cheese', 'Pepper'
]
selected_ingredients = st.multiselect('Available Ingredients:', ingredients)

if st.button('🔍 Generate Recipes'):
    if selected_ingredients:
        st.info("⏳ Generating recipes...")
        recipes = generate_recipes(selected_ingredients)
        if recipes:
            for idx, recipe_text in enumerate(recipes, 1):
                title, ingr, steps = parse_recipe(recipe_text)
                recipe_box = f"""
                <div style="border:2px solid #ccc; border-radius:10px; padding:15px; margin-bottom:20px; background-color:#28282B;">
                    <h4>📖 Recipe {idx}: {title}</h4>
                    <h5>🥕 Ingredients</h5>
                    <p>{ingr}</p>
                    <h5>🧑‍🍳 Cooking Steps</h5>
                    <p>{steps}</p>
                </div>
                """
                st.markdown(recipe_box, unsafe_allow_html=True)
        else:
            st.error("❌ No recipes generated. Please try again.")
    else:
        st.warning("⚠️ Please select at least one ingredient.")
