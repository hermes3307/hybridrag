#!/bin/bash

echo "🚀 Complete Face Vector Database Pipeline"
echo "========================================="
echo ""
echo "This will run all 6 steps sequentially:"
echo "1. Setup ChromaDB"
echo "2. Verify installation"
echo "3. Collect face data"
echo "4. Embed into database"
echo "5. Inspect database"
echo "6. Test semantic search"
echo ""

read -p "Do you want to run all steps? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "👋 Pipeline cancelled. You can run individual steps manually."
    echo "See README_STEPS.md for details."
    exit 0
fi

echo ""
echo "🎬 Starting complete pipeline..."
echo ""

# Run each step
for step in {1..6}; do
    script="${step}_*.sh"
    script_file=$(ls ${script} 2>/dev/null | head -1)

    if [ -f "$script_file" ]; then
        echo "⏳ Running $script_file..."
        ./"$script_file"

        if [ $? -eq 0 ]; then
            echo "✅ Step $step completed successfully!"
        else
            echo "❌ Step $step failed!"
            echo "Check the output above for errors."
            exit 1
        fi

        echo ""
        if [ $step -lt 6 ]; then
            read -p "Press Enter to continue to step $((step+1))..."
            echo ""
        fi
    else
        echo "❌ Script for step $step not found!"
        exit 1
    fi
done

echo "🎉 ALL STEPS COMPLETED SUCCESSFULLY!"
echo ""
echo "🎯 Your face vector database is now ready for use!"
echo ""
echo "Next steps:"
echo "• Run 'python3 face_database.py' for interactive search"
echo "• Run 'python3 test_face_system.py' for comprehensive testing"
echo "• See README_face_system.md for full documentation"