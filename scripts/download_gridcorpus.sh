#preparing for download
# sh download_gridcorpus.sh "y"
mkdir "gridcorpus"
cd "gridcorpus"
mkdir "raw" "audio" "video"
cd "raw" && mkdir "audio" "video" "align"
for i in $(seq 1 34)
do
    printf "\n\n------------------------- Downloading $i th speaker -------------------------\n\n"
    #download the audio of the ith speaker
    cd "audio" && curl "http://spandh.dcs.shef.ac.uk/gridcorpus/s$i/audio/s$i.tar" > "s$i.tar" && cd ..
    cd "video" && curl "http://spandh.dcs.shef.ac.uk/gridcorpus/s$i/video/s$i.mpg_vcd.zip" > "s$i.zip" && cd ..
    cd "align" && curl "http://spandh.dcs.shef.ac.uk/gridcorpus/s$i/align/s$i.tar" > "s$i.tar" && cd ..
    if [ $1 == "y" ]
    then
        unzip -q "video/s$i.zip" -d "../video"
        tar -xf "audio/s$i.tar" -C "../audio"
        tar -xf "align/s$i.tar" -C "../"
    fi
done
