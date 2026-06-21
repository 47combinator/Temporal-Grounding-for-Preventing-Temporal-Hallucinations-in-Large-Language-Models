"""
Cypher query templates for temporal fact retrieval from the Neo4j knowledge graph.

All templates use the Event Node Reification schema:
    (Entity)-[:ACTOR]->(Event)-[:TARGET]->(Entity)
where Event nodes hold: event_id, predicate, timestamp (Neo4j date), source.

Templates use ``$param`` placeholders for parameterised Neo4j queries.  No
string interpolation should be used at call sites -- always pass parameters
via the driver's parameter mechanism to prevent injection.
"""

# ---------------------------------------------------------------------------
# Entity at a specific time or within a date range
# ---------------------------------------------------------------------------

ENTITY_AT_TIME: str = """
MATCH (actor:Entity)-[:ACTOR]->(e:Event)-[:TARGET]->(target:Entity)
WHERE (actor.name = $entity_name OR target.name = $entity_name)
  AND e.timestamp >= date($start_date)
  AND e.timestamp <= date($end_date)
RETURN actor.name  AS subject,
       e.predicate AS predicate,
       target.name AS object,
       toString(e.timestamp) AS timestamp,
       e.event_id  AS event_id
ORDER BY e.timestamp
LIMIT $limit
"""

# ---------------------------------------------------------------------------
# Events involving a pair of entities (optional date range)
# ---------------------------------------------------------------------------

ENTITY_PAIR: str = """
MATCH (actor:Entity)-[:ACTOR]->(e:Event)-[:TARGET]->(target:Entity)
WHERE (
        (actor.name = $entity_a AND target.name = $entity_b)
        OR
        (actor.name = $entity_b AND target.name = $entity_a)
      )
  AND ($start_date IS NULL OR e.timestamp >= date($start_date))
  AND ($end_date   IS NULL OR e.timestamp <= date($end_date))
RETURN actor.name  AS subject,
       e.predicate AS predicate,
       target.name AS object,
       toString(e.timestamp) AS timestamp,
       e.event_id  AS event_id
ORDER BY e.timestamp
LIMIT $limit
"""

# ---------------------------------------------------------------------------
# Events before a given date for an entity
# ---------------------------------------------------------------------------

EVENTS_BEFORE: str = """
MATCH (actor:Entity)-[:ACTOR]->(e:Event)-[:TARGET]->(target:Entity)
WHERE (actor.name = $entity_name OR target.name = $entity_name)
  AND e.timestamp < date($reference_date)
RETURN actor.name  AS subject,
       e.predicate AS predicate,
       target.name AS object,
       toString(e.timestamp) AS timestamp,
       e.event_id  AS event_id
ORDER BY e.timestamp DESC
LIMIT $limit
"""

# ---------------------------------------------------------------------------
# Events after a given date for an entity
# ---------------------------------------------------------------------------

EVENTS_AFTER: str = """
MATCH (actor:Entity)-[:ACTOR]->(e:Event)-[:TARGET]->(target:Entity)
WHERE (actor.name = $entity_name OR target.name = $entity_name)
  AND e.timestamp > date($reference_date)
RETURN actor.name  AS subject,
       e.predicate AS predicate,
       target.name AS object,
       toString(e.timestamp) AS timestamp,
       e.event_id  AS event_id
ORDER BY e.timestamp
LIMIT $limit
"""

# ---------------------------------------------------------------------------
# Events of a specific predicate type in a date range
# ---------------------------------------------------------------------------

PREDICATE_SEARCH: str = """
MATCH (actor:Entity)-[:ACTOR]->(e:Event)-[:TARGET]->(target:Entity)
WHERE e.predicate = $predicate
  AND ($start_date IS NULL OR e.timestamp >= date($start_date))
  AND ($end_date   IS NULL OR e.timestamp <= date($end_date))
RETURN actor.name  AS subject,
       e.predicate AS predicate,
       target.name AS object,
       toString(e.timestamp) AS timestamp,
       e.event_id  AS event_id
ORDER BY e.timestamp
LIMIT $limit
"""

# ---------------------------------------------------------------------------
# All events for an entity (no time filter)
# ---------------------------------------------------------------------------

ENTITY_ALL_EVENTS: str = """
MATCH (actor:Entity)-[:ACTOR]->(e:Event)-[:TARGET]->(target:Entity)
WHERE actor.name = $entity_name OR target.name = $entity_name
RETURN actor.name  AS subject,
       e.predicate AS predicate,
       target.name AS object,
       toString(e.timestamp) AS timestamp,
       e.event_id  AS event_id
ORDER BY e.timestamp
LIMIT $limit
"""

# ---------------------------------------------------------------------------
# Exact fact verification -- check if a specific quadruple exists
# ---------------------------------------------------------------------------

FACT_VERIFICATION: str = """
MATCH (actor:Entity)-[:ACTOR]->(e:Event)-[:TARGET]->(target:Entity)
WHERE actor.name  = $subject
  AND e.predicate = $predicate
  AND target.name = $object
  AND ($timestamp IS NULL OR toString(e.timestamp) = $timestamp)
RETURN actor.name  AS subject,
       e.predicate AS predicate,
       target.name AS object,
       toString(e.timestamp) AS timestamp,
       e.event_id  AS event_id
ORDER BY e.timestamp
LIMIT $limit
"""
