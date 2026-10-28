// --- Data models ---
class Cypher {
  query string
}

class Answer {
  question string
  answer string
  used_talk_ids string[] @description("The talk_id values, exactly as they appear in square brackets in the context, of the talks the answer actually drew on. Empty when the answer drew on none.")
  from_general_knowledge bool @description("true when the answer rests on the domain knowledge rather than on any talk or event in the context")
}

// --- Functions ---

function RAGText2Cypher(schema: string, question: string, now: string) -> Cypher {
  client GeminiFlashChat
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
    - GIVES_TALK has a `date` property, but NEVER filter by it — for questions about the archive,
      always query all data regardless of any timeframe mentioned (e.g. "last 2 years", "recent",
      "latest"). The conference schedule below is the one exception.
    - Talk.video, Talk.heysummit, Talk.transcript, Talk.description_source, Event.url and
      IS_DESCRIBED_BY.source are provenance fields: they say where a value came from. Do not
      filter on them unless the question asks where something came from.
    - Talk.starts_at and Talk.ends_at (ISO date and time in London, e.g. "2026-11-11T09:30:00") and
      Talk.stage (the room or track, e.g. "Day 1: Rapid Insights Track - Ludgate Suite") are the
      programme's schedule, blank for talks HeySummit does not list.

    THE CONFERENCE SCHEDULE:
    The question is asked at a kiosk at the conference. NOW: {{ now }}
    When the question is about the conference itself — what is on now, next, today, tomorrow,
    this morning or afternoon, what to go and see, or when or where a talk is — query the current
    conference's talks (the Event named in NOW) by their schedule. Compare times as strings in
    the same ISO form as NOW: a talk is on now when t.starts_at <= NOW AND t.ends_at > NOW, and
    still to come when t.starts_at > NOW. "This afternoon" is 12:00 to 18:00 of today's date.
    Return t.talk_id, t.title, t.starts_at, t.ends_at and t.stage, ORDER BY t.starts_at, and
    LIMIT 10. Combine a topic with OR across tags, title and description as elsewhere.
    A question that asks for talks on a topic without mentioning the conference ("talks about
    RAG") is about the archive as well: do not filter it by time.

    EXAMPLES:
    Q: What is Connected Data?
    A: MATCH (e:Event) RETURN e.name, e.description

    Q: What technologies does Connected Data cover?
    A: MATCH (c:Category) RETURN c.name AS technology UNION MATCH (t:Talk)-[:IS_DESCRIBED_BY]->(tag:Tag) WITH tag.keyword AS technology, count(*) AS cnt ORDER BY cnt DESC LIMIT 15 RETURN technology

    Q: What are the most popular topics in the last 2 years?
    A: MATCH (t:Talk)-[:IS_DESCRIBED_BY]->(tag:Tag) RETURN tag.keyword, count(*) AS cnt ORDER BY cnt DESC LIMIT 15

    Q: Who are the speakers whose talks have the tag 'knowledge graph'?
    A: MATCH (s:Speaker)-[:GIVES_TALK]->(t:Talk)-[:IS_DESCRIBED_BY]->(tag:Tag) WHERE LOWER(tag.keyword) CONTAINS 'knowledge graph' RETURN DISTINCT s.name

    Q: Give me a summary of Paco Nathan's talks.
    A: MATCH (s:Speaker)-[:GIVES_TALK]->(t:Talk) WHERE LOWER(s.name) CONTAINS 'paco nathan' OPTIONAL MATCH (t)-[:IS_DESCRIBED_BY]->(tag:Tag) RETURN t.talk_id, t.title, t.description, collect(tag.keyword) AS tags

    Q: What did Paco Nathan talk about?
    A: MATCH (s:Speaker)-[:GIVES_TALK]->(t:Talk)-[:IS_DESCRIBED_BY]->(tag:Tag) WHERE LOWER(s.name) CONTAINS 'paco nathan' RETURN t.talk_id, t.title, collect(tag.keyword) AS tags

    Q: What Connected Data technologies are relevant to transparency and explainability?
    A: MATCH (s:Speaker)-[:GIVES_TALK]->(t:Talk)-[:IS_DESCRIBED_BY]->(tag:Tag) WHERE LOWER(tag.keyword) CONTAINS 'transparen' OR LOWER(tag.keyword) CONTAINS 'explainab' OR LOWER(t.description) CONTAINS 'transparen' OR LOWER(t.description) CONTAINS 'explainab' RETURN DISTINCT t.talk_id, t.title, t.description, s.name

    Q: Who are contributors to follow on transparency and explainability?
    A: MATCH (s:Speaker)-[:GIVES_TALK]->(t:Talk)-[:IS_DESCRIBED_BY]->(tag:Tag) WHERE LOWER(tag.keyword) CONTAINS 'explainab' OR LOWER(tag.keyword) CONTAINS 'transparen' OR LOWER(tag.keyword) CONTAINS 'ontolog' OR LOWER(tag.keyword) CONTAINS 'semantic' OR LOWER(tag.keyword) CONTAINS 'owl' OR LOWER(tag.keyword) CONTAINS 'shacl' OR LOWER(t.description) CONTAINS 'explainab' OR LOWER(t.description) CONTAINS 'transparen' RETURN DISTINCT t.talk_id, s.name, t.title, collect(tag.keyword) AS tags

    Q: What resources are there on building enterprise knowledge graphs?
    A: MATCH (s:Speaker)-[:GIVES_TALK]->(t:Talk)-[:IS_DESCRIBED_BY]->(tag:Tag) WHERE LOWER(tag.keyword) CONTAINS 'enterprise' OR LOWER(t.title) CONTAINS 'enterprise' OR LOWER(t.description) CONTAINS 'enterprise' RETURN DISTINCT t.talk_id, t.title, t.description, s.name

    Q: I am building an enterprise knowledge graph. Should I use OWL or SHACL? What are the arguments for and against each, and who has given them?
    A: MATCH (s:Speaker)-[:GIVES_TALK]->(t:Talk)-[:IS_DESCRIBED_BY]->(tag:Tag) WHERE LOWER(tag.keyword) = 'owl' RETURN 'OWL' AS technology, t.talk_id, t.title, t.description, s.name UNION MATCH (s:Speaker)-[:GIVES_TALK]->(t:Talk)-[:IS_DESCRIBED_BY]->(tag:Tag) WHERE LOWER(tag.keyword) = 'shacl' RETURN 'SHACL' AS technology, t.talk_id, t.title, t.description, s.name
    NOTE: For technology comparison questions (X vs Y), the UNION query filters ONLY on the technology tags — never add extra WHERE clauses from the question preamble (e.g. "enterprise"). Retrieve all talks mentioning either technology so the answer has real content to compare.

    The schedule examples below are for NOW = Wednesday 11 November 2026, 10:10 London time
    (2026-11-11T10:10), at Connected Data London 2026; use the real NOW above.

    Q: What's on right now?
    A: MATCH (s:Speaker)-[:GIVES_TALK]->(t:Talk)-[:IS_PART_OF]->(e:Event) WHERE e.name = 'Connected Data London 2026' AND t.starts_at <= '2026-11-11T10:10' AND t.ends_at > '2026-11-11T10:10' RETURN t.talk_id, t.title, t.starts_at, t.ends_at, t.stage, collect(s.name) AS speakers ORDER BY t.starts_at LIMIT 10

    Q: Is there anything on LLMs or RAG later today?
    A: MATCH (s:Speaker)-[:GIVES_TALK]->(t:Talk)-[:IS_PART_OF]->(e:Event), (t)-[:IS_DESCRIBED_BY]->(tag:Tag) WHERE e.name = 'Connected Data London 2026' AND t.starts_at > '2026-11-11T10:10' AND t.starts_at < '2026-11-12' AND (LOWER(tag.keyword) CONTAINS 'llm' OR LOWER(tag.keyword) CONTAINS 'rag' OR LOWER(t.title) CONTAINS 'llm' OR LOWER(t.title) CONTAINS 'rag' OR LOWER(t.description) CONTAINS 'llm' OR LOWER(t.description) CONTAINS 'retrieval') RETURN DISTINCT t.talk_id, t.title, t.starts_at, t.ends_at, t.stage, s.name ORDER BY t.starts_at LIMIT 10

    Q: Where and when is the talk on ontologies without ontologists?
    A: MATCH (s:Speaker)-[:GIVES_TALK]->(t:Talk)-[:IS_PART_OF]->(e:Event) WHERE e.name = 'Connected Data London 2026' AND LOWER(t.title) CONTAINS 'ontologies without ontologists' RETURN t.talk_id, t.title, t.starts_at, t.ends_at, t.stage, s.name

    The question is:
    {{ question }}

    INSTRUCTIONS:
    1. Use only the provided node and relationship types and properties in the schema.
    2. When returning results, return property values rather than the entire node or relationship.
    3. Use LOWER() + CONTAINS for all string comparisons, never exact equality on text fields —
       except the Event name and the schedule times, compared exactly as in the examples.
    4. Each Tag node holds a single keyword. When searching multiple concepts, use OR across
       separate CONTAINS clauses — never AND on the same tag variable.
    5. Whenever a Talk is part of the answer, return t.talk_id alongside its other properties,
       so the answer can cite the talk. Never return only talk_id: include t.title too.
    6. A WHERE filters only the MATCH or OPTIONAL MATCH directly before it. Filter a speaker,
       talk or event in a WHERE right after the MATCH that binds it, before any OPTIONAL MATCH:
       a WHERE placed after an OPTIONAL MATCH filters nothing but the optional part, and the
       query returns every talk in the graph.

    {{ ctx.output_format }}
  "#
}

function RAGAnswerQuestion(question: string, context: string, grounding: string, now: string) -> Answer {
  client GeminiFlashChat
  prompt #"
    You are an AI assistant for the Connected Data conference knowledge base.
    You will be provided with a question and graph query results as context.
    Use the context as your primary source, but always supplement with the domain knowledge below
    when it is relevant. If the context is empty or insufficient, rely on the domain knowledge.
    Always strive to provide concise, helpful answers.
    For comparison questions (X vs Y, should I use X or Y): only cite arguments that are
    explicitly stated in the provided context. Do not add general knowledge trade-offs.
    If no explicit argument exists for one side, say so clearly.

    The context was retrieved from the knowledge graph, and GROUNDING says what it holds:
    "talks" — talks, each introduced by its id in square brackets, e.g. [t-1a2b3c4d];
    "events" — only editions of the conference; "general" — nothing that answers the question.
    Base the answer on the context, and anchor it: name the talks and the speakers the answer
    draws on in the answer text itself, as a well-read colleague would ("Veronika Heimsbakk's
    'Modelling regulation requirements using SHACL' argues that…"). In used_talk_ids, list the
    ids of those talks, and only those. Set from_general_knowledge to true when the answer rests
    on the domain knowledge below rather than on the context. Two things stay out of the answer
    text: the bracketed ids themselves, and a trailing "Sources" list — the application renders
    the references from used_talk_ids. Do not use the word "context".

    The question is asked at a kiosk at the conference. NOW: {{ now }}
    A talk with "scheduled:" is on the current programme, and the bracket after it says where it
    stands at NOW ("on now", "starts in 25 min", "ended"). When recommending one, say when and
    where in plain words ("at 11:35 in the Warwick Suite, in 25 minutes"). Never suggest going to
    a talk that has ended, nor one starting on a later day as if it were today. Talks from earlier
    editions are recordings to watch afterwards, not sessions to attend.

    DOMAIN KNOWLEDGE:
    - "Connected Data" is an umbrella term for technologies that treat connections in data as
      first-class citizens, including Knowledge Graphs, Graph Databases, Graph Analytics,
      Semantic Technology, and Graph AI.
    - The term was coined and popularized by the Connected Data London conference. As an event,
      Connected Data first started as a series of London-based Meetups in 2016. That same year,
      the first Connected Data London conference was held.
    - The knowledge graph in this system covers the conference's programme from 2019 onward
      (Connected Data London 2019, Knowledge Connexions 2020, Connected Data World 2021, the
      meetups, Connected Data London 2024 and 2025) and the programme of Connected Data London
      2026 — it does not contain the full history.
    - As a technological space, the strands of Connected Data (e.g. graph theory) have existed
      for much longer; they were brought together conceptually by the Connected Data London event.
    - The conference covers the full stack of graph and semantic technologies and their
      application in AI, data science, and knowledge management.

    {{ _.role("user") }}
    QUESTION: {{ question }}
    GROUNDING: {{ grounding }}
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
        - talk_id: string
        - title: string
        - category: string
        - url: string
        - description: string
        - type: string
        - video: string
        - heysummit: string
        - transcript: string
        - description_source: string
        - starts_at: string
        - stage: string
        - ends_at: string
    - Speaker
        - name: string
    - Event
        - name: string
        - description: string
        - url: string
    - Category
        - name: string

    Edge properties:
    - GIVES_TALK
        - date: date
    - IS_DESCRIBED_BY
        - source: string
    "#
    question "Who are the speakers whose talks have the tag 'rdf'? Return the speaker names as a numbered list."
    now "Wednesday 11 November 2026, 10:10 London time (2026-11-11T10:10). The current conference, Connected Data London 2026, runs 11–12 November 2026."
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
    grounding "talks"
    now "Wednesday 11 November 2026, 10:10 London time (2026-11-11T10:10)."
  }
}

test RAGMissingContext {
  functions [RAGAnswerQuestion]
  args {
    question "What did Paco Nathan say about RDF?"
    context ""
    grounding "general"
    now "Wednesday 11 November 2026, 10:10 London time (2026-11-11T10:10)."
  }
  @@assert(flagged, {{ this.from_general_knowledge == true }})
}
