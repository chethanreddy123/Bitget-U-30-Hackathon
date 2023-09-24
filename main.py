import streamlit as st
import google.generativeai as palm

palm.configure(api_key='AIzaSyA1fu-ob27CzsJozdr6pHd96t5ziaD87wM')

def generate_text(model, prompt):
    completion = palm.generate_text(
        model=model,
        prompt=prompt,
        temperature=0,
        # The maximum length of the response
        max_output_tokens=800,
    )
    return completion.result

def main():
    st.title("Streamlit Chat App with PALM API")

    models = [m for m in palm.list_models() if 'generateText' in m.supported_generation_methods]
    model = models[0].name

    # Chat input
    user_input = st.text_area("You:", value="", height=100)

    if st.button("Generate Response"):
        if user_input.strip() != "":
            response = generate_text(model, user_input)
            st.text("Bot:")
            st.write(response)

if __name__ == "__main__":
    main()


