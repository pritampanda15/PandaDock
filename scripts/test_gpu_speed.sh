#!/bin/bash
# Quick GPU speed test - should complete in ~30 seconds (not 3 hours!)

echo "=========================================="
echo "🚀 GPU Speed Test - After Optimization"
echo "=========================================="
echo ""
echo "Testing: cuda_monte_carlo algorithm"
echo "Expected time: 20-40 seconds"
echo "Your previous time: 11,837 seconds (3.3 hours!)"
echo ""
echo "Starting test..."
echo ""

START_TIME=$(date +%s)

PYTHONPATH=. python3 -m pandadock.docking_cli dock \
  --receptor preprocessing/intermediate/receptor_prepared.pdb \
  --ligand preprocessing/intermediate/ligands/ligand.sdf \
  --algorithm cuda_monte_carlo \
  --output-dir test_gpu_speed_fixed \
  --num-iterations 1000 \
  --batch-size 1000

END_TIME=$(date +%s)
RUNTIME=$((END_TIME - START_TIME))

echo ""
echo "=========================================="
echo "✅ Test Complete!"
echo "=========================================="
echo "Runtime: $RUNTIME seconds"
echo ""

if [ $RUNTIME -lt 120 ]; then
    echo "🎉 SUCCESS! GPU is now FAST!"
    echo "   Expected: 20-40 seconds"
    echo "   Actual: $RUNTIME seconds"
    echo "   Speedup vs previous: ~$((11837 / RUNTIME))× faster!"
    echo ""
    echo "✅ All optimizations working correctly!"
elif [ $RUNTIME -lt 600 ]; then
    echo "⚠️  PARTIAL SUCCESS - Faster but not optimal"
    echo "   Expected: 20-40 seconds"
    echo "   Actual: $RUNTIME seconds"
    echo "   Still has some bottlenecks"
else
    echo "❌ STILL SLOW - Optimizations may not be working"
    echo "   Expected: 20-40 seconds"
    echo "   Actual: $RUNTIME seconds"
    echo "   Check error logs above"
fi

echo ""
echo "Next steps:"
echo "  1. If fast (< 2 min): Run full benchmark"
echo "  2. If still slow: Check error messages above"
echo ""
