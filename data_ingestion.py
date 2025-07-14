import os
import pickle
from dotenv import load_dotenv
import requests
from urllib.parse import urljoin, urlparse

load_dotenv()

from langchain_community.document_loaders import WebBaseLoader
import bs4

# Define o USER_AGENT a partir da variável de ambiente ou usa um valor padrão
USER_AGENT = os.getenv("USER_AGENT", "CloudWalkBot/1.0")

# Função para extrair links de uma página HTML
# Adicionada para evitar links duplicados e garantir que são URLs absolutas
def extract_links(html_content, base_url):
    """
    Extrai todos os links internos de uma página HTML, garantindo que são URLs absolutas.
    """
    soup = bs4.BeautifulSoup(html_content, 'html.parser')
    links = set()
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        full_url = urljoin(base_url, href)

        parsed_base = urlparse(base_url)
        parsed_full = urlparse(full_url)

        if parsed_full.netloc == parsed_base.netloc or \
           parsed_full.netloc.endswith(f".{parsed_base.netloc}") or \
           parsed_base.netloc.endswith(f".{parsed_full.netloc}"):

            if full_url.startswith("http") and "#" not in full_url:
                links.add(full_url)
    return list(links)

def load_cloudwalk_data():
    """
    Loads content from CloudWalk websites using LangChain's WebBaseLoader
    and performs a basic link crawling, or loads from existing pickle if available.
    """
    docs_pkl_path = "cloudwalk_documents.pkl"

    # Se o arquivo .pkl já existe, carrega os dados pré-scrapeados
    # e filtra documentos vazios ou com pouco conteúdo
    if os.path.exists(docs_pkl_path):
        print(f"'{docs_pkl_path}' found. Loading pre-scraped data from file to save time.")
        with open(docs_pkl_path, "rb") as f:
            all_documents = pickle.load(f)
        print(f"Loaded {len(all_documents)} documents from '{docs_pkl_path}'.")
        # Filtrar documentos vazios ou com pouco conteúdo mesmo ao carregar de PKL
        initial_doc_count = len(all_documents)
        all_documents = [doc for doc in all_documents if len(doc.page_content.strip()) > 100]
        print(f"Filtered to {len(all_documents)} non-empty documents.")
        if all_documents:
            print("\n--- Sample of first loaded document content ---")
            print(all_documents[0].page_content[:500])
            print("---------------------------------------------\n")
        return all_documents



    initial_urls = [
        "https://www.cloudwalk.io",
        "https://www.infinitepay.io/tap",
        "https://www.reclameaqui.com.br/empresa/cloudwalk/", # Exemplo: Adicionar a URL aqui
        "https://www.reclameaqui.com.br/empresa/infinitepay/", # Exemplo: Adicionar a URL aqui
    ]

    visited_urls = set()
    urls_to_visit = list(initial_urls)
    all_documents = []

    print("Starting web crawling process (no existing data found)...")
    while urls_to_visit:
        current_url = urls_to_visit.pop(0)

        if current_url in visited_urls:
            continue

        print(f"Visiting: {current_url}")
        visited_urls.add(current_url)

        try:
            headers = {"User-Agent": USER_AGENT}
            response = requests.get(current_url, headers=headers, timeout=10)
            response.raise_for_status()

            html_content = response.text
            new_links = extract_links(html_content, current_url)

            for link in new_links:
                if link not in visited_urls and link not in urls_to_visit:
                    urls_to_visit.append(link)

            loader = WebBaseLoader(
                web_paths=[current_url],
                bs_kwargs={
                    "parse_only": bs4.SoupStrainer(
                        ["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "span", "div", "a"]
                    )
                },
                requests_kwargs={"headers": {"User-Agent": USER_AGENT}}
            )
            docs = loader.load()
            all_documents.extend(docs)

        except requests.exceptions.RequestException as e:
            print(f"Error visiting {current_url}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred with {current_url}: {e}")

    # Filtrar documentos vazios ou com pouco conteúdo no final
    initial_doc_count = len(all_documents)
    all_documents = [doc for doc in all_documents if len(doc.page_content.strip()) > 100]
    print(f"Loaded {initial_doc_count} documents. Filtered to {len(all_documents)} non-empty documents.")

    if all_documents:
        print("\n--- Sample of first loaded document content ---")
        print(all_documents[0].page_content[:500])
        print("---------------------------------------------\n")

    return all_documents

if __name__ == "__main__":
    cloudwalk_docs = load_cloudwalk_data()
    # Salva os documentos SEMPRE, para que a próxima execução possa carregá-los
    with open("cloudwalk_documents.pkl", "wb") as f:
        pickle.dump(cloudwalk_docs, f)
    print("Data loading complete. Proceed to chunking and embedding.")