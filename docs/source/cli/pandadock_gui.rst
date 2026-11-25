pandadock-gui - PandaDock Studio Launcher
=========================================

Launches **PandaDock Studio**, the local Apple-designed web GUI for PandaDock v5.0 with guided 4-step docking setup, 3D/2D molecular visualization (powered by **3Dmol.js**, **RDKit**, and **PandaLens**), 300-DPI plots, embedded A4 vector PDF viewer, editorial HTML reports, and live console streaming.

Synopsis
--------

.. code-block:: bash

   pandadock-gui [OPTIONS]
   # Equivalent alias:
   pandadock gui [OPTIONS]

Options
-------

.. list-table::
   :header-rows: 1
   :widths: 24 14 62

   * - Option
     - Default
     - Description
   * - ``--host TEXT``
     - ``127.0.0.1``
     - Local network address to bind the FastAPI/Uvicorn server.
   * - ``--port INTEGER``
     - ``8765``
     - Browser port number (e.g. ``http://127.0.0.1:8765``). Use ``--port 8785`` if 8765 is occupied.
   * - ``--workspace DIR``
     - ``.``
     - Directory scanned for input structures and completed docking results shown in the Recent Results sidebar.
   * - ``--no-browser``
     - off
     - Do not automatically open a browser tab (useful on remote SSH / cluster sessions).

Examples
--------

.. code-block:: bash

   # Launch PandaDock Studio in your current docking workspace
   pandadock-gui --port 8765 --workspace .

   # Headless / SSH tunnel launch
   pandadock gui --host 127.0.0.1 --port 8785 --workspace ./projects --no-browser

See :doc:`../guide/studio_gui` for a complete step-by-step walkthrough of the Studio workspace.
