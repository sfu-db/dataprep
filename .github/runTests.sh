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
                test_file_name=$PWD/tests/${parentdir}/${name}_test.py
                echo ${test_file_name}
                python ./dataprep/tests/eda/test.py
        fi
done
