echo $PWD
for file in $1
do
        echo $file
        filename="$(basename -- $file)"
        extension="${filename##*.}"
        if [ ${extension} == "py" ] && [[ $filename != *"test"* ]]
        then
                parentdir="$(basename "$(dirname "$file")")"
                echo ${parentdir}
                name="${filename%.*}"
                test_file_name=./dataprep/tests/${parentdir}/${name}_test.py
                echo ${test_file_name}
                poetry run pytest test_file_name
        fi
done
poetry run pytest ./dataprep/tests/eda/test.py
poetry run python ./dataprep/tests/eda/test_plot.py
poetry run python ./dataprep/tests/eda/test_plot_correlation.py
poetry run python ./dataprep/tests/eda/test_plot_missing.py

