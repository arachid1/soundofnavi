Welcome to soundofnavi's documentation!
========================================

soundofnavi has a dual purpose. It serves as a data modelling package for biomedical (lung) audio data, in synergy with a machine learning pipeline emphasizing scalability and interpretability.
The library structures any lung sound dataset into a neat object oriented design, where concepts of Datasets, Patients, Recordings and Slices are tied together for audio loading, preparation, augmentation, and advanced analysis.
The fun comes in with a machine learning framework applicable on multiple tasks (i.e., pneumonia, crackles/wheezes) with extensive diagnosis not only on the fixed audio, but across any desired critera (i.e., by dataset, patients of a certain demographic) over time.   

.. toctree::
   :caption: Contents:

.. toctree::
   :caption: Intro to soundofnavi:

   installation/introduction
   installation/getting_started

.. toctree::
   :caption: Working with soundofnavi:

   user_info/structure

.. toctree::
   :caption: API reference/dev info:

   dev_info/api_reference
   dev_info/ci_testing



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`