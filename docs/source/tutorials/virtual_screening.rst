Virtual Screening & Series SAR Tutorial
=======================================

Learn when to use ``pandadock screen`` (diverse compound libraries) versus ``pandadock series`` (congeneric medicinal-chemistry analog series).

High-Throughput Library Screening (``pandadock screen``)
--------------------------------------------------------

``pandadock screen`` precomputes receptor affinity grids once and batches ligands onto CUDA/MPS/CPU:

.. code-block:: bash

   pandadock screen -r receptor.pdb -l library.sdf -g box.json \
                    --device cuda --n-chains 128 --max-batch 64 \
                    --top 50 -o screen_hits/

Congeneric Analog Series SAR (``pandadock series``)
---------------------------------------------------

When optimizing close analogs sharing a chemical scaffold, tether the Maximum Common Substructure (MCS) core onto your reference crystal pose and generate a publishable A4 SAR brochure PDF:

.. code-block:: bash

   pandadock series -r receptor.pdb -l analogs.smi \
                    --reference-pose parent_xray.sdf --core-tolerance 0.75 \
                    --center 126.5 119.2 146.1 --box 22 22 22 -o series_sar/
