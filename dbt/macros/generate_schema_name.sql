{% macro generate_schema_name(custom_schema_name, node) -%}

    {#-
        Override dbt's default schema naming.

        Default behaviour prepends target.schema to the custom schema name,
        producing e.g. "staging_marts" when target.schema = "staging" and
        the model has +schema: marts.  That breaks the scorer which expects
        relations to live in the raw "marts" schema.

        This override uses the custom schema name directly when one is given,
        and falls back to target.schema for models without an explicit schema.
    -#}

    {%- if custom_schema_name is none -%}
        {{ target.schema }}
    {%- else -%}
        {{ custom_schema_name | trim }}
    {%- endif -%}

{%- endmacro %}
