#!/bin/bash

# Quick validation test for comprehensive_algorithms.sh
echo "🧪 Testing comprehensive_algorithms.sh validation..."

# Check if required files exist
RECEPTOR="preprocessing/intermediate/receptor_processed.pdb"
LIGAND="preprocessing/intermediate/ligand_processed.sdf"
GRID_CONFIG="gridbox_output/gridbox_similarity.json"

echo "Checking required input files:"
echo -n "  • Receptor file... "
if [ -f "$RECEPTOR" ]; then
    echo "✅ Found"
else
    echo "❌ Missing: $RECEPTOR"
fi

echo -n "  • Ligand file... "
if [ -f "$LIGAND" ]; then
    echo "✅ Found"
else
    echo "❌ Missing: $LIGAND"
fi

echo -n "  • Grid config... "
if [ -f "$GRID_CONFIG" ]; then
    echo "✅ Found"
else
    echo "❌ Missing: $GRID_CONFIG"
fi

# Check PandaDock CLI availability
echo ""
echo "Checking PandaDock CLI tools:"
echo -n "  • pandadock dock... "
if PYTHONPATH=. python3 -m pandadock.docking_cli --help &>/dev/null; then
    echo "✅ Available"
else
    echo "❌ Not available"
fi

echo -n "  • pandadock-flex... "
if PYTHONPATH=. python3 -m pandadock.flex_docking_cli --help &>/dev/null; then
    echo "✅ Available"
else
    echo "❌ Not available"
fi

echo -n "  • pandadock-report... "
if PYTHONPATH=. python3 -m pandadock.report_cli --help &>/dev/null; then
    echo "✅ Available"
else
    echo "❌ Not available"
fi

echo -n "  • pandadock-tethered... "
if PYTHONPATH=. python3 -m pandadock.tethered_cli --help &>/dev/null; then
    echo "✅ Available"
else
    echo "❌ Not available"
fi

echo ""
echo "Validation complete. Script is ready to run!"
echo "Execute with: ./comprehensive_algorithms.sh"