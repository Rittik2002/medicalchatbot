from setuptools import find_packages,setup

setup(
    name='medicalchatbot',
    version='0.0.1',
    author='Rittik Raj',
    author_email='rittikrajchauhan123@gmail.com',
    install_requires=["langchain","langchain-community","sentence-transformers","streamlit","python-dotenv","PyPDF2","pinecone-client"],
    packages=find_packages()
)