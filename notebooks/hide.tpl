{%- extends 'basic.tpl' -%}
<!-- hide.tpl uses collapse_show collapse_hide hide_input hide_output -->
{% block codecell %}
    {{ "{% raw %}" }}
    {{ super() }}
    {{ "{% endraw %}" }}
{% endblock codecell %}

{% block input_group -%}
{%- if cell.metadata.collapse_show -%}
    <details class="description" open>
      <summary class="btn btn-sm" data-open="Hide Code" data-close="Show Code"></summary>
        <p>{{ super() }}</p>
    </details>
{%- elif cell.metadata.collapse_hide -%}
    <details class="description">
      <summary class="btn btn-sm" data-open="Hide Code" data-close="Show Code"></summary>
        <p>{{ super() }}</p>
    </details>
{%- elif cell.metadata.hide_input or nb.metadata.hide_input or cell.metadata.remove_cell -%}
{%- else -%}
    {{ super()  }}
{%- endif -%}
{% endblock input_group %}

{% block output_group -%}
{%- if cell.metadata.hide_output or cell.metadata.remove_cell -%}
{%- else -%}
    {{ super()  }}
{%- endif -%}
{% endblock output_group %}

{% block output_area_prompt %}
{%- if cell.metadata.hide_input or nb.metadata.hide_input or cell.metadata.remove_cell-%}
   <div class="prompt"> </div>
{%- else -%}
    {{ super()  }}
{%- endif -%}
{% endblock output_area_prompt %}