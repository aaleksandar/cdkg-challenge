// "Talks for you": a conference visitor holds their LinkedIn profile (on their
// phone) or their badge up to the demo laptop's camera, and gets the talks in
// the graph that match what they work on.
//
// Two calls, deliberately separate. ReadVisitor turns the photo into a profile
// and a handful of the graph's own tags; the talks are then chosen by a fixed
// Cypher query, not by the model, so the cards appear as soon as this returns.
// WhyForYou writes the one-line reasons afterwards, while the cards are already
// on screen.
//
// Both bind to GeminiFlashLive: the same model as everything else, with a
// retry policy sized for someone standing in front of the screen.

class VisitorProfile {
  found bool @description("true ONLY if the photo shows a person's LinkedIn profile, conference badge or business card with a readable name or job title")
  name string? @description("The person's name exactly as shown, or null")
  headline string? @description("Their LinkedIn headline or job title, verbatim, or null")
  company string? @description("Their current company or organisation, or null")
  about string? @description("Up to two sentences from the visible 'About' or summary text, verbatim, or null")
  interests string[] @description("Three to six short phrases for what this person most likely works on, inferred from their headline, company and about text")
  tags string[] @description("Five to twelve entries copied EXACTLY from the TAGS list that best match the interests; never a tag that is not in the list")
}

function ReadVisitor(photo: image, tags: string) -> VisitorProfile {
  client GeminiFlashLive
  prompt #"
    {{ _.role("user") }}
    You are the welcome screen of a demo at Connected Data London, a conference
    about knowledge graphs, graph databases, semantic technology and graph AI.
    A visitor is holding up their LinkedIn profile on a phone, their conference
    badge, or a business card to the camera.

    {{ photo }}

    Read who they are, then pick the topics from the knowledge graph they would
    most want to hear about.

    Rules:
    - Only read what is visible. If the photo shows no profile, badge or card,
      or nothing is legible, set found=false and leave everything empty.
    - Copy name, headline and company verbatim. Never guess a name.
    - If a word is too small or blurred to read with certainty, leave that
      field null rather than filling in a plausible word. A wrong name on the
      screen is worse than no name.
    - interests: what this person works on, inferred from their role and what
      their company does. Prefer concrete topics ("clinical data integration",
      "fraud detection") over generic ones ("technology").
    - tags: choose ONLY from the TAGS list below, copied character for character.
      Prefer specific tags over broad ones like "knowledge graphs" or "ai", but
      include a broad one when nothing specific fits. Always return at least
      five when found=true.

    TAGS (most used first):
    {{ tags }}

    {{ ctx.output_format }}
  "#
}

class Pick {
  talk_id string
  why string @description("One sentence, addressed to the visitor as 'you', linking the talk to their work. Under 25 words.")
}

class Recommendation {
  intro string @description("One friendly sentence greeting the visitor by first name (if known) and saying what the picks have in common. Under 30 words.")
  picks Pick[]
}

function WhyForYou(visitor: string, talks: string) -> Recommendation {
  client GeminiFlashLive
  prompt #"
    {{ _.role("user") }}
    A visitor at Connected Data London has been matched to talks from the
    conference's knowledge graph. Tell them, briefly and warmly, why each talk
    is for them.

    VISITOR:
    {{ visitor }}

    TALKS (each under its id):
    {{ talks }}

    Rules:
    - One pick per talk above, using its id exactly. Do not add talks.
    - Each "why" draws on what the talk is about (its title, tags, description)
      and what the visitor does. No hype, no exclamation marks, no flattery.
    - Never claim anything about the visitor that is not in VISITOR.

    {{ ctx.output_format }}
  "#
}

test WhyForYouGreetsAndLinks {
  functions [WhyForYou]
  args {
    visitor "name: Jane Doe | headline: Senior Ontologist | company: AstraZeneca | interests: biomedical ontologies, data integration"
    talks #"
      [t-1] SHACL in practice — Ada Lovelace — tags: shacl, owl, data validation
      [t-2] Graph neural networks for drug discovery — Alan Turing — tags: graph neural networks, drug discovery
    "#
  }
}
