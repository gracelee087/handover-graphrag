"""Single source of truth for the graph schema.

Three kinds of schema live here (roadmap step 02):
  1. STRUCTURE_*   : nodes/relationships loaded from mail headers, regex and SOP
                     structure with plain Cypher (no LLM).   -> load_structure.py, parse_sop.py
  2. EXTRACTION_*  : entity/relation types the LLM may extract from mail bodies
                     and SOP clause text.                    -> extract_mail.py, extract_sop.py
  3. CONSTRAINTS / VECTOR_INDEXES : Cypher DDL.               -> load_structure.py, retrievers.py

Every other module imports from here. Nothing else defines a label name.
"""

# ---------------------------------------------------------------------------
# 1. Structure layer (deterministic, from world.yaml / headers / regex)
# ---------------------------------------------------------------------------
STRUCTURE_NODES = {
    "Person": ["id", "name", "email", "title", "is_internal"],
    "Organization": ["id", "name", "type"],  # type: forwarder|carrier|broker|shipper|trucker|terminal
    "Shipment": [
        "id", "mbl", "hbl", "container", "vessel", "voyage",
        "pol", "pod", "eta", "ata", "free_time_days", "status", "has_incident",
    ],
    "Thread": ["id", "subject"],
    "Email": ["id", "message_id", "subject", "date", "body"],
    "Attachment": ["id", "filename"],
    "SOPDocument": ["id", "title", "version", "effective_date", "filename"],
    "Section": ["id", "number", "title"],
    "Clause": ["id", "number", "title", "text"],
}

STRUCTURE_RELS = [
    ("Person", "WORKS_AT", "Organization"),
    ("Person", "SENT", "Email"),
    ("Email", "TO", "Person"),
    ("Email", "CC", "Person"),
    ("Email", "IN_THREAD", "Thread"),
    ("Email", "ABOUT", "Shipment"),
    ("Email", "HAS_ATTACHMENT", "Attachment"),
    ("Email", "REPLIES_TO", "Email"),
    ("Shipment", "CARRIED_BY", "Organization"),
    ("Shipment", "CLEARED_BY", "Organization"),
    ("Shipment", "CONSIGNEE", "Organization"),
    ("Shipment", "SHIPPER", "Organization"),
    ("SOPDocument", "HAS_SECTION", "Section"),
    ("Section", "HAS_CLAUSE", "Clause"),
    ("Section", "NEXT_SECTION", "Section"),
    ("Clause", "NEXT_CLAUSE", "Clause"),
]

# ---------------------------------------------------------------------------
# 2. Extraction layer (what the LLM may produce; passed to SimpleKGPipeline)
# ---------------------------------------------------------------------------
EXTRACTION_NODE_TYPES = [
    {"label": "Decision", "description": "A business decision made in the correspondence",
     "properties": [{"name": "summary", "type": "STRING"}, {"name": "date", "type": "STRING"}]},
    {"label": "Task", "description": "An action item someone has to do",
     "properties": [{"name": "summary", "type": "STRING"}, {"name": "due", "type": "STRING"},
                    {"name": "status", "type": "STRING"}]},
    {"label": "Issue", "description": "A problem or incident with a shipment",
     "properties": [{"name": "summary", "type": "STRING"}, {"name": "kind", "type": "STRING"}]},
    {"label": "Deadline", "description": "A dated cut-off (free time expiry, filing deadline)",
     "properties": [{"name": "date", "type": "STRING"}, {"name": "what", "type": "STRING"}]},
    # SOP side
    {"label": "Rule", "description": "A normative requirement stated in an SOP clause",
     "properties": [{"name": "summary", "type": "STRING"}, {"name": "modality", "type": "STRING"}]},
    {"label": "Role", "description": "A job role named in an SOP (Import Operator, Team Lead...)",
     "properties": [{"name": "name", "type": "STRING"}]},
    # shared, resolved against structure layer by name/id
    {"label": "Person", "properties": [{"name": "name", "type": "STRING"}]},
    {"label": "Organization", "properties": [{"name": "name", "type": "STRING"}]},
    {"label": "Shipment", "properties": [{"name": "id", "type": "STRING"}]},
]

EXTRACTION_REL_TYPES = [
    {"label": "DECIDED_BY"},
    {"label": "ASSIGNED_TO"},
    {"label": "RAISED_BY"},
    {"label": "CONCERNS"},        # Decision/Task/Issue/Deadline -> Shipment
    {"label": "RESOLVES"},        # Decision -> Issue
    {"label": "HAS_DEADLINE"},    # Task/Shipment -> Deadline
    {"label": "REQUIRES"},        # Rule -> Role  (SOP)
    {"label": "APPLIES_TO_ROLE"}, # Rule -> Role
]

EXTRACTION_PATTERNS = [
    ("Decision", "DECIDED_BY", "Person"),
    ("Decision", "CONCERNS", "Shipment"),
    ("Decision", "RESOLVES", "Issue"),
    ("Task", "ASSIGNED_TO", "Person"),
    ("Task", "CONCERNS", "Shipment"),
    ("Task", "HAS_DEADLINE", "Deadline"),
    ("Issue", "RAISED_BY", "Person"),
    ("Issue", "CONCERNS", "Shipment"),
    ("Deadline", "CONCERNS", "Shipment"),
    ("Rule", "APPLIES_TO_ROLE", "Role"),
]

# Link produced in link_sop.py (vector similarity + LLM confirm), not by extraction
LINK_RELS = [
    ("Issue", "GOVERNED_BY", "Clause"),
    ("Decision", "GOVERNED_BY", "Clause"),
    ("Email", "CITES", "Clause"),
]

# neo4j-graphrag lexical-graph defaults (verified in library source, v1.24.0)
CHUNK_LABEL = "Chunk"
DOCUMENT_LABEL = "Document"
CHUNK_TEXT_PROP = "text"
CHUNK_EMBEDDING_PROP = "embedding"
REL_FROM_DOCUMENT = "FROM_DOCUMENT"
REL_FROM_CHUNK = "FROM_CHUNK"
REL_NEXT_CHUNK = "NEXT_CHUNK"

# ---------------------------------------------------------------------------
# 3. Cypher DDL
# ---------------------------------------------------------------------------
CONSTRAINTS = [
    f"CREATE CONSTRAINT {label.lower()}_id IF NOT EXISTS FOR (n:{label}) REQUIRE n.id IS UNIQUE"
    for label in STRUCTURE_NODES
] + [
    "CREATE CONSTRAINT email_message_id IF NOT EXISTS FOR (n:Email) REQUIRE n.message_id IS UNIQUE",
]

# Embedding model is fixed here so every script embeds with the same dimensions.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # sentence-transformers, local, 384 dims
EMBEDDING_DIM = 384

VECTOR_INDEXES = {
    # name: (label, property)
    "chunk_embedding": (CHUNK_LABEL, CHUNK_EMBEDDING_PROP),
    "clause_embedding": ("Clause", "embedding"),
}

# LLM (Anthropic). Key is read from ANTHROPIC_API_KEY by the SDK.
LLM_MODEL = "claude-opus-5-5"
LLM_MAX_TOKENS = 4096


def all_labels() -> set[str]:
    return set(STRUCTURE_NODES) | {n["label"] for n in EXTRACTION_NODE_TYPES} | {
        CHUNK_LABEL, DOCUMENT_LABEL}


def all_rel_types() -> set[str]:
    return ({r[1] for r in STRUCTURE_RELS}
            | {r["label"] for r in EXTRACTION_REL_TYPES}
            | {r[1] for r in LINK_RELS}
            | {REL_FROM_DOCUMENT, REL_FROM_CHUNK, REL_NEXT_CHUNK})
