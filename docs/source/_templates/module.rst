{{ fullname }}
{{ underline }}

.. automodule:: {{ fullname }}

   {% block functions %}
   {% if functions %}

   Functions
   ---------

   {% for item in functions %}

   .. autofunction:: {{ item }}

   .. include:: backreferences/{{fullname}}.{{item}}.examples

   .. raw:: html

	       <div class="sphx-glr-clear"></div>

   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block classes %}
   {% if classes %}

   Classes
   -------

   {% for item in classes %}
   .. autoclass:: {{ item }}
      :members:

   .. include:: backreferences/{{fullname}}.{{item}}.examples

   .. raw:: html

	       <div class="sphx-glr-clear"></div>

   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block exceptions %}
   {% if exceptions %}

   Exceptions
   ----------

   .. autosummary::
   {% for item in exceptions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}