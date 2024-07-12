# AI Travel Agent Project

## Overview

This project is designed to provide comprehensive travel itineraries using the power of language models and web data extraction. It combines various tools to gather, process, and generate detailed travel plans based on user queries. The project focuses on integrating information about places to visit, events, flight prices, and car rentals.

## Purpose

The primary purpose of this project is to create an automated system that can:

1. Search the web for relevant travel information.
2. Load and process web content to extract useful data.
3. Generate a detailed and customized travel itinerary.
4. Provide accurate pricing information.

## Features

-   **Web Searching**: Uses search tools to find the latest and most relevant information.
-   **Document Loading**: Extracts data from specific web pages.
-   **Text Splitting and Embedding**: Processes large documents and creates embeddings for efficient retrieval.
-   **Vector Store**: Stores and retrieves documents based on their content similarity.
-   **Custom Prompt Handling**: Generates tailored responses using prompt templates and sequences.

## Libraries Used

-   `langchain`: A framework for building applications with language models.
-   `langchain_openai`: Integration with OpenAI's language models.
-   `langchain_community`: Community-contributed tools and components.
-   `langchain.agents`: Tools for creating and managing agents that interact with language models.
-   `langchain_community.document_loaders`: Utilities for loading and processing web documents.
-   `langchain_community.vectorstores`: Vector storage solutions for document retrieval.
-   `langchain_text_splitters`: Tools for splitting text into manageable chunks.
-   `langchain_core.prompts`: Prompt templates for structured interactions.
-   `langchain_core.runnables`: Components for creating and running sequences of actions.
-   `bs4` (Beautiful Soup): A library for web scraping and parsing HTML/XML documents.

## How It Works

1. **Research Agent**:

    - Uses the language model and search tools to find relevant web content based on the user's query.
    - Returns the gathered context information.

2. **Load Data**:

    - Loads and parses web pages using `WebBaseLoader` and `BeautifulSoup`.
    - Splits documents into chunks and creates embeddings for these chunks.
    - Stores the processed documents in a vector store (`Chroma`).

3. **Get Relevant Documents**:

    - Retrieves documents from the vector store that are most relevant to the user's query.

4. **Supervisor Agent**:

    - Uses a prompt template to generate a detailed travel itinerary.
    - Combines web context, relevant documents, and user input to create a comprehensive response.

5. **Get Response**:
    - Coordinates the entire process: research, document retrieval, and itinerary generation.
    - Produces the final response with the travel plan.

## Usage

To run the project, execute the main script. It will:

1. Perform a web search for relevant travel information.
2. Load and process web pages.
3. Retrieve relevant documents from the vector store.
4. Generate a detailed travel itinerary based on the provided query.

## Contributions

Contributions are welcome! Please fork this repository and submit pull requests for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---
