# Streamlit generate

You will need to read and runài necessary entire notebook to have full understanding about process
Here are request:

Still keep to main flow of Content Baserd Suggestion notebok
Add addtional fancy, beatiful chart into EDA part.
Use only 'Company Name', 'Company overview', 'Our key skills'
Ensure have 2 features (is function, after than can be reuse in streamlit): recommed from a company name, and recommend from a text input
Add more model mentioned in yeucau1.ipynb. then compare the arracy or similiary by chart (fancy, nice chart)
Entire notebook should have good documentation
Aslo edit readme.md after documentation in notebook
You must restart and re-run entire notebook at the end, make sure code is in good structure and no bug. Reading the result should explain and give insight too.

# Change streamlit

You will act as me - a data scientist who is working on a Streamlit application for a company recommendation system. The app will be structured into three main sections, each with its own navigation bar and functionality. You will need to implement the following changes and enhancements to the existing Streamlit app based on the provided requirements:
Here are request:

- Streamlit app now will use 3 differents navigation bar, the first is "Content-Based Company Similarity System", the second is "Recommendation Modeling System". Each of them will have similar to right now.
- The third session will be "About", which contains information about author. Referent author in thinhdao.typ
- Add emails to the "Team Members": Đào Tuấn Thịnh is daotuanthinh@gmail.com and Trương Văn Lê is truongvanle999@gmail.com
- Use the picture of this https://itviec.com/assets/logo_black_text-04776232a37ae9091cddb3df1973277252b12ad19a16715f4486e603ade3b6a4.png for panel (maybe use or not)

The "Content-Based Company Similarity System":
    - In notebook Content Based Suggestion.ipynb, there is now have more method to pick. So add those method to the configuration
    - The configuration part will now appear within page. Recommendation Method now will appear close to Get Recommendation
    - Re-write the business object to be more clear, and add more information about the business object, use the file README_Enhanced.md and README.md for reference.
    - There now will be 4 tabs: 
        - "About" which will based on both file Readme and information in Content Based Suggestion.ipynb, here will have information about the business object, the data, and the methods.
        - "Company Recommendation", which will provide recommendations based on company names.
        - "Text Recommendation", which will provide recommendations based on text input.
        - "EDA and Visualization", which will provide EDA and visualization of the data, all model comparation and all charts in notebook Content Based Suggestion.ipynb. Try input all methods and compare the results, then visualize the results in a fancy way, good structure.

The "Recommendation Modeling System": mainly based on the notebook Recommendation Modeling.ipynb
    - The configuration section should now be displayed directly on the page, positioned near the "Predict Recommendation" button for easy access.
    - Allow users to select different machine learning models and relevant parameters as part of the configuration.
    - Add clear documentation and descriptions for each available model and configuration option, referencing both the notebook and README files for details.
    - Organize this section with tabs:
            - "About": Summarize the business objective, data, and modeling approaches, using information from the notebook and README files.
            - "Predict Recommendation": Enable users to input data and receive recommendations, with model selection and configuration options visible.
            - "Model Comparison": Present results and comparisons of all implemented models using visually appealing and informative charts.
            - "EDA and Visualization": Display exploratory data analysis and visualizations relevant to the recommendation models, ensuring clarity and insight.

Note:

- Ensure that the Streamlit app is well-structured, with clear navigation and user-friendly interfaces.
- Maintain a consistent design and layout across all sections.
- All will be written in English, with clear and concise explanations.
- Edit and run notebook and rerun if necessary to ensure all features are functional and well-documented.
- After edit and change app.py, need to run the Streamlit app to ensure everything works as expected with .venv environment already set up.
- Clean up file app_backup, app_new, app.py, maintain only one app.py file, other should be removed if there are no more use.
- Clean up config.py, create_preprocess_data.py, FIXES_SUMMARY.md, if no use.
- Clean up retrain_models.py, retrain_models if no use.
- Ensure that the final Streamlit app is fully functional, with all features working as intended and no bugs present.
- Combine README_Enhanced.md and README.md into one README.md file, ensuring it contains all necessary information about the project, including setup instructions, usage examples, and detailed explanations of the business objectives, data requirements, and model performance.
- You will need to read and run all if necessary entire notebook to have full understanding about process
- All notebooks and app.py should be well-documented, with clear explanations of each step and function, consistent with each others