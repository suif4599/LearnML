SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

for file in $(ls $SCRIPT_DIR | grep -v 'clear.sh'); do
    echo "Removing $file"
    rm -rf "${SCRIPT_DIR}/${file}"
done