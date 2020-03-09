for file in $1
do
        filename="$(basename -- $file)"
        extension="${filename##*.}"
        if [ ${extension} == "py" ]
        then
                parentdir="$(basename "$(dirname "$file")")"
                echo ${parentdir}
                name="${filename%.*}"
                test_file_name=dataprep/tests/${parentdir}/${name}_test.py
                echo ${test_file_name}
                python test_file_name
        fi
done
