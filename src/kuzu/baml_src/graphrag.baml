// --- Data models ---
class Cypher {
  query string
}

class Answer {
  question string
  answer string
}

// --- Functions ---

function RAGText2Cypher(schema: string, question: string) -> Cypher {
  client Gemini2Flash
  prompt #"
    You are an expert in translating natural language questions into Cypher statements.
    You will be provided with a question and a graph schema.
    Use only the provided relationship types and properties in the schema to generate a Cypher
    statement.
    The Cypher statement could retrieve nodes, relationships, or both.
    Do not include any explanations or apologies in your responses.
    Do not respond to any questions that might ask anything else than for you to construct a
    Cypher statement.
    Answer in no more than 5 sentences

    {{ _.role("user") }}
    Task: Generate a Cypher statement to query a graph database.

    {{ schema }}

    DOMAIN NOTES:
    - Event nodes represent editions of the Connected Data conference (e.g. "Connected Data World 2021").
      "Connected Data" refers to this conference, not a talk title or tag keyword.
    - Category nodes represent the three topic areas: Knowledge Graphs, Graph AI, Semantic Technology.
    - Tag nodes hold keywords extracted from talk transcripts (always lowercase, one keyword per node).
    - GIVES_TALK has a `date` property, but NEVER filter by date — always query all data regardless
      of any timeframe mentioned in the question (e.g. "last 2 years", "recent", "latest").

    EXAMPLES:
    Q: What is Connected Data?
    A: MATCH (e:Event) RETURN e.name, e.description

    Q: What technologies does Connected Data cover?
    A: MATCH (c:Category) RETURN c.name

    Q: What are the most popular topics in the last 2 years?
    A: MATCH (t:Talk)-[:IS_DESCRIBED_BY]->(tag:Tag) RETURN tag.keyword, count(*) AS cnt ORDER BY cnt DESC LIMIT 15

    Q: Who are the speakers whose talks have the tag 'knowledge graph'?
    A: MATCH (s:Speaker)-[:GIVES_TALK]->(t:Talk)-[:IS_DESCRIBED_BY]->(tag:Tag) WHERE LOWER(tag.keyword) CONTAINS 'knowledge graph' RETURN DISTINCT s.name

    Q: What did Paco Nathan talk about?
    A: MATCH (s:Speaker)-[:GIVES_TALK]->(t:Talk)-[:IS_DESCRIBED_BY]->(tag:Tag) WHERE LOWER(s.name) CONTAINS 'paco nathan' RETURN t.title, collect(tag.keyword) AS tags

    Q: What Connected Data technologies are relevant to transparency and explainability?
    A: MATCH (s:Speaker)-[:GIVES_TALK]->(t:Talk)-[:IS_DESCRIBED_BY]->(tag:Tag) WHERE LOWER(tag.keyword) CONTAINS 'transparen' OR LOWER(tag.keyword) CONTAINS 'explainab' OR LOWER(t.description) CONTAINS 'transparen' OR LOWER(t.description) CONTAINS 'explainab' RETURN DISTINCT t.title, t.description, s.name

    Q: Who are contributors to follow on transparency and explainability?
    A: MATCH (s:Speaker)-[:GIVES_TALK]->(t:Talk)-[:IS_DESCRIBED_BY]->(tag:Tag) WHERE LOWER(tag.keyword) CONTAINS 'explainab' OR LOWER(tag.keyword) CONTAINS 'transparen' OR LOWER(tag.keyword) CONTAINS 'ontolog' OR LOWER(tag.keyword) CONTAINS 'semantic' OR LOWER(tag.keyword) CONTAINS 'owl' OR LOWER(tag.keyword) CONTAINS 'shacl' OR LOWER(t.description) CONTAINS 'explainab' OR LOWER(t.description) CONTAINS 'transparen' RETURN DISTINCT s.name, t.title, collect(tag.keyword) AS tags

    Q: What resources are there on building enterprise knowledge graphs?
    A: MATCH (s:Speaker)-[:GIVES_TALK]->(t:Talk)-[:IS_DESCRIBED_BY]->(tag:Tag) WHERE LOWER(tag.keyword) CONTAINS 'enterprise' OR LOWER(t.title) CONTAINS 'enterprise' OR LOWER(t.description) CONTAINS 'enterprise' RETURN DISTINCT t.title, t.description, s.name

    The question is:
    {{ question }}

    INSTRUCTIONS:
    1. Use only the provided node and relationship types and properties in the schema.
    2. When returning results, return property values rather than the entire node or relationship.
    3. Use LOWER() + CONTAINS for all string comparisons, never exact equality on text fields.
    4. Each Tag node holds a single keyword. When searching multiple concepts, use OR across
       separate CONTAINS clauses — never AND on the same tag variable.

    {{ ctx.output_format }}
  "#
}

function RAGAnswerQuestion(question: string, context: string) -> Answer {
  client Gemini2Flash
  prompt #"
    You are an AI assistant for the Connected Data conference knowledge base.
    You will be provided with a question and graph query results as context.
    Use the context as your primary source. If the context is empty or insufficient,
    use your general knowledge about the Connected Data conference and community to answer.
    Always strive to provide concise, helpful answers.

    {{ _.role("user") }}
    QUESTION: {{ question }}
    RELEVANT CONTEXT: {{ context }}

    {{ ctx.output_format }}

    RESPONSE:
  "#
}


test Text2Cypher1 {
  functions [RAGText2Cypher]
  args {
    schema #"
    ALWAYS RESPECT THE EDGE DIRECTIONS:
    ---
    (:Talk) -[:IS_DESCRIBED_BY]-> (:Tag)
    (:Speaker) -[:GIVES_TALK]-> (:Talk)
    (:Talk) -[:IS_PART_OF]-> (:Event)
    (:Talk) -[:IS_CATEGORIZED_AS]-> (:Category)
    ---

    Node properties:
    - Tag
        - keyword: string
    - Talk
        - title: string
        - category: string
        - url: string
        - description: string
        - type: string
    - Speaker
        - name: string
    - Event
        - name: string
        - description: string
    - Category
        - name: string

    Edge properties:
    - GIVES_TALK
        - date: date
    "#
    question "Who are the speakers whose talks have the tag 'rdf'? Return the speaker names as a numbered list."
  }
  @@assert(has_return_clause, {{ 'RETURN' in this.query }})
}

test RAGAnswer1 {
  functions [RAGAnswerQuestion]
  args {
    question "Who are the speakers whose talks have the tag 'rdf'? Return the speaker names as a numbered list."
    context #"
        ['Marcus Nölke', 'Veronika Heimsbakk', 'Dave Duggal', 'Kurt Cagle', 'Sebastian Schmidt', 'Veronique Moore', 'Atanas Kiryakov', 'Ora Lassila']
    "#
  }
}

test RAGMissingContext {
  functions [RAGAnswerQuestion]
  args {
    question "What did Paco Nathan say about RDF?"
    context ""
  }
}
