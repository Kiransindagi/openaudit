from setuptools import setup, find_packages

setup(
    name="openaudit",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "pydantic>=2.0.0",
        "openai>=1.0.0",
        "requests>=2.31.0",
        "python-dotenv>=1.0.0",
    ],
)
