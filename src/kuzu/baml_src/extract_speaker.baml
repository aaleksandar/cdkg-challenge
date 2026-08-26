// Recover the speaker from a talk's YouTube description when the deterministic
// parser cannot. The parser reads titles ("Talk | Speaker | Event") and a small
// set of description phrasings ("A talk by X"); channels are not consistent, and
// a speaker is often just a name on its own line above a bio.
//
// This is extraction, not inference. The graph is built from the CSV verbatim,
// so a wrong Speaker is worse than a blank one: the function is told to return
// found=false whenever the description does not actually attribute the talk, and
// the caller additionally refuses anything that does not look like a person's
// name. "Probably the CEO mentioned in paragraph three" must never reach a row.

class SpeakerAttribution {
  found bool @description("true ONLY if the description explicitly attributes this talk to a named person")
  speaker string? @description("The person's name alone: no title, no role, no company, no honorific, no URL. 'Atanas Kiryakov. CEO & Founder, Ontotext' -> 'Atanas Kiryakov'. Join co-speakers with ' & '.")
  evidence string? @description("The exact sentence or line the name was taken from, copied verbatim, so a curator can check it")
}

function ExtractSpeaker(title: string, description: string) -> SpeakerAttribution {
  client GeminiFlash
  prompt #"
    You are reading the YouTube description of a recorded conference talk to find
    out who gave it.

    Title:
    {{ title }}

    Description:
    {{ description }}

    Return the name of the person who DELIVERED this talk.

    Rules, in order of importance:
    - If the description does not clearly state who gave the talk, set found=false
      and leave speaker null. A blank field is correct and useful; a guess is not.
    - Never infer a speaker from the topic, the company, or someone merely
      mentioned or quoted in the talk. Only a person the description presents as
      the presenter counts.
    - The conference organiser, the host channel ("Connected Data", "Connected
      Data London") and sponsors are NOT the speaker.
    - A speaker is usually introduced by name on its own line, often followed by a
      role and company and then a short biography, or by a phrase such as
      "a talk by", "presented by", "with", or "speaker:".
    - Return the name only. Strip titles (Dr, Prof), roles, companies, emails and
      URLs. If two or more people presented together, join them with " & ".
    - Ignore the promotional footer advertising future conferences.

    {{ ctx.output_format }}
  "#
}

test SpeakerOnItsOwnLineAboveABio {
  functions [ExtractSpeaker]
  args {
    title "Talk to your data: leverage open schema & knowledge for efficient self-service insights | CDL24"
    description #"
      Discover how to democratize data access and eliminate the traditional
      bottleneck where business users must wait weeks for technical experts.

      --

      Atanas Kiryakov. CEO & Founder, Ontotext

      Atanas is the founder and CEO of Ontotext– Semantic Web pioneer and leading
      developer of semantic technology and knowledge graphs.

      --

      Welcome to Connected Data London's #ThrowbackThursday
      Connected Data London 2024 has been announced!
    "#
  }
}

test NoSpeakerStatedAtAll {
  functions [ExtractSpeaker]
  args {
    title "Knowledge Graph Conference highlights"
    description #"
      A round-up of the best moments from this year's event. Featuring clips from
      several sessions on ontologies, SHACL and graph machine learning.

      Subscribe for more, and check CDL24 for the full programme.
    "#
  }
}

test MustNotReturnTheCompanyOrThePanel {
  functions [ExtractSpeaker]
  args {
    title "The future of graph databases | Panel Discussion"
    description #"
      Neo4j, Ontotext and Stardog debate where graph databases are heading.
      Hosted by Connected Data London.
    "#
  }
}
