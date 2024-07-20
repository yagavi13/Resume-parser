import streamlit as st 
import pdfplumber
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time

st.title("RESUME FILTER")

uploadedJD = st.file_uploader("Upload Job Description", type="pdf")

uploadedResumes = st.file_uploader("Upload resumes (multiple PDFs allowed)",type="pdf", accept_multiple_files=True)

click = st.button("ANALYSE")

def process_pdf(pdf_file):
    if pdf_file is not None:
        try:
            with pdfplumber.open(pdf_file) as pdf:
                page = pdf.pages[0]
                text = page.extract_text()
                return text
        except Exception as e:
            st.error(f"Error occurred while processing PDF: {e}")
            return None
    else:
        return None

def getResult(JD_txt, resume_txt):
    if JD_txt is not None and resume_txt is not None:
        content = [JD_txt, resume_txt]
        
        # Custom stop words list
        custom_stop_words = ["is", "the", "was", "and", "other", "common", "grammatical", "terms"]
        
        cv = CountVectorizer(stop_words=custom_stop_words)
        matrix = cv.fit_transform(content)
        similarity_matrix = cosine_similarity(matrix)
        match = similarity_matrix[0][1] * 100
        return round(match, 2)
    else:
        return None
if click:
    with st.spinner('Processing...'):
        time.sleep(10) 
        
        if uploadedJD is not None and uploadedResumes is not None:
            job_description = process_pdf(uploadedJD)
            
            suitable_resumes = []
            not_suitable_resumes = []

            for uploadedResume in uploadedResumes:
                resume = process_pdf(uploadedResume)
                if resume:
                    match = getResult(job_description, resume)
                    if match is not None:
                        if match > 50:
                            suitable_resumes.append((uploadedResume.name, match))
                        else:
                            not_suitable_resumes.append((uploadedResume.name, match))
                    else:
                        st.warning(f"Unable to calculate match percentage for {uploadedResume.name}.")
                else:
                    st.warning("Please upload valid resume files.")

            if suitable_resumes:
                st.subheader("Suitable Resumes:")
                for resume_name, match_score in suitable_resumes:
                    st.write(f"- {resume_name} (Match Percentage: {match_score}%)")

            if not_suitable_resumes:
                st.subheader("Not Suitable Resumes:")
                for resume_name, match_score in not_suitable_resumes:
                    st.write(f"- {resume_name} (Match Percentage: {match_score}%)")
        else:
            st.warning("Please upload both job description and resume files.")