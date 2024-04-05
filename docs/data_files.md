# Data Files

A project dataset comprises a series of `.jsonl` files and a single `.json` metadata file. These files collectively encapsulate relational knowledge relations. **TREx** dataset is included for demonstration purposes.

## Relation data: JSONL Files

Each `.jsonl` file represents relational data points (triple). The format for each line in these files consists of:

- **predicate_id**: A unique identifier (Wikidata) for the relation.
- **sub_id**: A unique identifier (Wikidata) for the subject entity.
- **sub_label**: A readable label or name for the subject entity.
- **obj_id**: A unique identifier (Wikidata) for the object entity.
- **obj_label**: A readable label or name for the object entity.

### Example Instance:

```json
{
  "predicate_id":"P30",
  "sub_id":"Q3108669",
  "sub_label":"Lavoisier Island",
  "obj_id":"Q51",
  "obj_label":"Antarctica"
}
```

## Metadata JSON File

The `metadata_relations.json` file provides contextual and auxiliary information about each relation (identified by the `predicate_id`) available in the `.jsonl` files. It specifies:

- **templates:** A list of sentence templates that can be used to form human-readable statements from triples in a given relation.
- **answer_space:** A list of possible answers that can be associated with the particular relation (set of all answers).

### Example Metadata:

```json
{
  "P30": {
    "templates": ["[X] is located in [Y]."],
    "answer_space": ["Asia", "Antarctica", "Africa", "Europe", "Americas", "Oceania"]
  }
}
```

