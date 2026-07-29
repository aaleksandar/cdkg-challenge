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
  client GeminiFlash
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
    A: MATCH (c:Category) RETURN c.name AS technology UNION MATCH (t:Talk)-[:IS_DESCRIBED_BY]->(tag:Tag) WITH tag.keyword AS technology, count(*) AS cnt ORDER BY cnt DESC LIMIT 15 RETURN technology

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

    Q: I am building an enterprise knowledge graph. Should I use OWL or SHACL? What are the arguments for and against each, and who has given them?
    A: MATCH (s:Speaker)-[:GIVES_TALK]->(t:Talk)-[:IS_DESCRIBED_BY]->(tag:Tag) WHERE LOWER(tag.keyword) = 'owl' RETURN 'OWL' AS technology, t.title, t.description, s.name UNION MATCH (s:Speaker)-[:GIVES_TALK]->(t:Talk)-[:IS_DESCRIBED_BY]->(tag:Tag) WHERE LOWER(tag.keyword) = 'shacl' RETURN 'SHACL' AS technology, t.title, t.description, s.name
    NOTE: For technology comparison questions (X vs Y), the UNION query filters ONLY on the technology tags — never add extra WHERE clauses from the question preamble (e.g. "enterprise"). Retrieve all talks mentioning either technology so the answer has real content to compare.

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
  client GeminiFlash
  prompt #"
    You are an AI assistant for the Connected Data conference knowledge base.
    You will be provided with a question and graph query results as context.
    Use the context as your primary source, but always supplement with the domain knowledge below
    when it is relevant. If the context is empty or insufficient, rely on the domain knowledge.
    Always strive to provide concise, helpful answers.
    For comparison questions (X vs Y, should I use X or Y): only cite arguments that are
    explicitly stated in the provided context. Do not add general knowledge trade-offs.
    If no explicit argument exists for one side, say so clearly.

    DOMAIN KNOWLEDGE:
    - "Connected Data" is an umbrella term for technologies that treat connections in data as
      first-class citizens, including Knowledge Graphs, Graph Databases, Graph Analytics,
      Semantic Technology, and Graph AI.
    - The term was coined and popularized by the Connected Data London conference. As an event,
      Connected Data first started as a series of London-based Meetups in 2016. That same year,
      the first Connected Data London conference was held.
    - The knowledge graph in this system only covers talks from 2020 onward (Knowledge Connexions
      2020, Connected Data World 2021, and a 2024 meetup) — it does not contain the full history.
    - As a technological space, the strands of Connected Data (e.g. graph theory) have existed
      for much longer; they were brought together conceptually by the Connected Data London event.
    - The conference covers the full stack of graph and semantic technologies and their
      application in AI, data science, and knowledge management.

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
