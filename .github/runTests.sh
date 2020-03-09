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
                python test_file_name
        fi
done
python ./dataprep/tests/eda/test.py
python ./dataprep/tests/eda/test_plot.py
python ./dataprep/tests/eda/test_plot_correlation.py
python ./dataprep/tests/eda/test_plot_missing.py

