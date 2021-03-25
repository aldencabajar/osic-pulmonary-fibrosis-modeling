#!/bin/bash

# make sure that you have your kaggle.json file in hand 
#and that it is located in root/.kaggle!
while getopts d: flag
do
	case "${flag}" in
		d) save_dir=${OPTARG};;
	esac

done

if ! [[-d "$save_dir"]]
then 
	mkdir "$save_dir"
fi
	

kaggle competitions download m5-forecasting-uncertainty -p "$save_dir" --force

cd "$save_dir"
unzip *.zip
rm *.zip










