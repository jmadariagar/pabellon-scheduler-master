from django import template

register = template.Library()


@register.simple_tag(takes_context=True)
def set_global_context(context, key, value):
    """
    Sets a value to the global template context, so it can
    be accessible across blocks.
    Note that the block where the global context variable is set must appear
    before theE other blocks using the variable IN THE BASE TEMPLAT.  The order
    of the blocks in the extending template is not important.
    Usage::
        {% extends 'base.html' %}
        {% block first %}
            {% set_global_context 'foo' 'bar' %}
        {% endblock %}
        {% block second %}
            {{ foo }}
        {% endblock %}
    """
    context.dicts[0][key] = value
    return ''