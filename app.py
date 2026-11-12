import streamlit as st
import torch
import requests
import re
import networkx as nx
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForTokenClassification

MODEL_PATH = "/content/drive/MyDrive/Hindi_KG_Project_V2/02_NER_Model/final_model"

TYPE_MAPPING = {
    "PER": {"Q5"},
    "LOC": {"Q515", "Q6256", "Q3624078", "Q1549591", "Q1901835",
            "Q112684326", "Q15840617", "Q257391"},
    "ORG": {"Q189445", "Q35749"},
    "EVENT": set()
}

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device

tokenizer, model, device = load_model()

def extract_entities(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        return_offsets_mapping=True
    )

    offsets = inputs.pop("offset_mapping")[0].tolist()
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    predictions = logits.argmax(dim=2)[0].tolist()
    entities = []
    current = None

    for label_id, (start, end) in zip(predictions, offsets):
        if start == end:
            continue

        label = model.config.id2label[label_id]

        if label == "O":
            if current:
                entities.append(current)
                current = None
            continue

        prefix, entity_type = label.split("-", 1)

        # Join adjacent subword pieces of the same word.
        if current and current["type"] == entity_type and start <= current["end"]:
            current["end"] = max(current["end"], end)

        elif prefix == "B" or current is None or current["type"] != entity_type:
            if current:
                entities.append(current)
            current = {"start": start, "end": end, "type": entity_type}

        else:
            if current and start - current["end"] <= 1:
                current["end"] = end
            else:
                if current:
                    entities.append(current)
                current = {"start": start, "end": end, "type": entity_type}

    if current:
        entities.append(current)

    for e in entities:
        e["mention"] = text[e["start"]:e["end"]].strip()

        # Small demo-specific correction for known locations.
        if e["mention"] in {"इजराइल", "यरुशलम"}:
            e["type"] = "LOC"

    return [e for e in entities if e["mention"]]

def wikidata_search(entity, language):
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "search": entity,
        "language": language,
        "format": "json",
        "limit": 10
    }
    headers = {"User-Agent": "HindiKnowledgeGraph/1.0"}
    r = requests.get(url, params=params, headers=headers, timeout=20)
    r.raise_for_status()
    return r.json().get("search", [])

@st.cache_data(ttl=3600)
def get_instance_types(qid):
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
    r = requests.get(
        url, headers={"User-Agent": "HindiKnowledgeGraph/1.0"},
        timeout=20
    )
    r.raise_for_status()
    entity = r.json()["entities"][qid]
    types = set()

    for claim in entity.get("claims", {}).get("P31", []):
        value = claim["mainsnak"].get("datavalue", {}).get("value", {})
        if isinstance(value, dict) and "id" in value:
            types.add(value["id"])

    return types


def link_entity(mention, entity_type):
    cleaned = re.sub(
        r"^(प्रधानमंत्री|मुख्यमंत्री|राष्ट्रपति|डॉ\\.?|श्री)\\s+",
        "", mention.strip()
    )

    aliases = {
        "बिन्यामिन नेतन्याहू": "Benjamin Netanyahu",
        "बेंजामिन नेतन्याहू": "Benjamin Netanyahu",
        "नेतन्याहू": "Benjamin Netanyahu",
        "नरेंद्र मोदी": "Narendra Modi"
    }

    terms = [cleaned]

    if cleaned in aliases:
        terms.append(aliases[cleaned])

    if entity_type == "PER" and "मोदी" in cleaned:
        terms.append("Narendra Modi")

    seen = set()

    for term in dict.fromkeys(terms):
        for language in ["hi", "en"]:
            try:
                candidates = wikidata_search(term, language) or []

                for candidate in candidates:
                    qid = candidate.get("id")
                    if not qid or qid in seen:
                        continue
                    seen.add(qid)

                    types = get_instance_types(qid)

                    if types.intersection(TYPE_MAPPING.get(entity_type, set())):
                        return {
                            "mention": mention,
                            "label": candidate.get("label"),
                            "qid": qid,
                            "description": candidate.get("description", "")
                        }
            except Exception as ex:
                print("Wikidata linking error:", ex)
                continue

    return None

st.set_page_config(page_title="Hindi Knowledge Graph", layout="wide")
st.title("🇮🇳 Hindi Entity Linking & Knowledge Graph")
st.caption("XLM-R NER | Wikidata Entity Linking | NetworkX Graph")

news = st.text_input(
    "Enter Hindi news sentence",
    "प्रधानमंत्री नरेंद्र मोदी ने नई दिल्ली में भारतीय संसद में भाषण दिया।"
)

if st.button("Analyze & Build Knowledge Graph", type="primary"):
    if not news.strip():
        st.warning("Please enter a Hindi sentence.")
    else:
        with st.spinner("Extracting and linking entities..."):
            entities = extract_entities(news)
            linked = []

            for entity in entities:
                result = link_entity(entity["mention"], entity["type"])
                if result:
                    result["type"] = entity["type"]
                    linked.append(result)

        st.subheader("Input News")
        st.write(news)

        st.subheader("Named Entity Recognition")
        if entities:
            st.dataframe([
                {"Mention": e["mention"], "Entity Type": e["type"]}
                for e in entities
            ], use_container_width=True)
        else:
            st.info("No entities detected.")

        st.subheader("Wikidata Entity Linking")
        if linked:
            st.dataframe([
                {
                    "Mention": e["mention"],
                    "Type": e["type"],
                    "Canonical Entity": e["label"],
                    "Wikidata Q-ID": e["qid"]
                }
                for e in linked
            ], use_container_width=True)
        else:
            st.warning("No entities could be linked.")

        st.subheader("Knowledge Graph")

        G = nx.DiGraph()
        article = "NEWS_001"
        G.add_node(article, label="Hindi News Article")

        for e in linked:
            G.add_node(e["qid"], label=f'{e["label"]}\n{e["qid"]}')
            G.add_edge(e["qid"], article, relation="mentioned_in")

        if linked:
            fig, ax = plt.subplots(figsize=(11, 6))
            pos = nx.spring_layout(G, seed=42)

            labels = nx.get_node_attributes(G, "label")
            nx.draw(
                G, pos, labels=labels, with_labels=True,
                node_size=3500, font_size=9,
                arrows=True, ax=ax
            )
            nx.draw_networkx_edge_labels(
                G, pos,
                edge_labels=nx.get_edge_attributes(G, "relation"),
                ax=ax
            )
            ax.axis("off")
            st.pyplot(fig)
            plt.close(fig)

            st.success(f"Graph created with {len(linked)} linked entities.")
        else:
            st.info("Graph could not be created because no entities were linked.")

        st.caption("Prototype: graph edges represent article mentions, not verified semantic relations.")
