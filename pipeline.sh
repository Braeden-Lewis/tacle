echo "(-:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::-)"
echo " ( \ .         88888888888        .d8888b.  888      8888888888      . / )"
echo "  ( \ \            888           d88P  Y88b 888      888            / / )"
echo "   ( ) )           888           888    888 888      888           ( ( )"
echo "  ( / /            888   8888b.  888        888      8888888        \ \ )"
echo "  ( \ \            888      88b  888        888      888            / / )"
echo "   ( ) )           888  .d888888 888    888 888      888           ( ( )"
echo "  ( / /            888  888  888 Y88b  d88P 888      888            \ \ )"
echo " ( / '             888  Y888888    Y88888P  88888888 8888888888      ' \ )"
echo "(-:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::-)"
echo -e "\n"
echo "                        ----- Document Info -----                        "
echo -e "\n"
echo "Package -- Tool-assisted Classification using Lexical Elements"
echo "Author -- Braeden Lewis"
echo "Language -- Python v3.10.1"
echo "(-:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::-)"
echo -e "\n"
echo "Generating run configurations..."
python -m create_config
return_val=$?
if $return_val==0; then
  echo "Configuration file created!"
  echo "(-:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::-)"
  continue
else
  echo "An unexpected error occurred. Ending run."
  exit 1
fi
echo -e "\n"
echo "Initiating data extraction..."
python -m code.extraction.run_extraction
return_val=$?
if $return_val==0; then
  echo "Extraction complete!"
  echo "(-:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::-)"
  continue
else
  echo "Data extraction failed. Ending run."
  exit 1
fi
echo -e "\n"
echo "Beginning natural language processing..."
python -m code.nlp.run_nlp
return_val=$?
if $return_val==0; then
  echo "Completed!"
  echo "(-:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::-)"
  continue
else
  echo "Natural language processing failed. Ending run."
  exit 1
fi
echo "(-:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::-)"
echo -e "\n"
echo "Evaluating machine learning models on dataset..."
python -m code.machine-learning.run_machine_learning
return_val=$?
if $return_val==0; then
  echo "Evaluation complete!"
  echo "Output files for machine learning can be found in the following directory:"
  echo "/usr/share/tacle/data/output/csv-files/"
  echo "(-:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::-)"
  continue
else
  echo "Machine learning evaluation failed. Ending run."
  exit 1
fi
