package main

import (


package main

import (
	"fmt"
	"log"
	"os"

	"github.com/kniren/gota/dataframe"
	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/evaluation"
	"github.com/sjwhitworth/golearn/knn"
)

func main() {
	// *** Loading data ***
	data, err := base.ParseCSVToInstances("Iris.csv", true)
	if err != nil {
		panic(err)
	}

	fmt.Println(data)


	*** KNN classifier ***
	cls := knn.NewKnnClassifier("euclidean", "linear", 2)	
	train, test := base.InstancesTrainTestSplit(data, 0.50)
	cls.Fit(train)

	// ** prediction **
	predictions, err := cls.Predict(test)
	if err != nil {
		panic(err)
	}
	fmt.Println(predictions)

	// *** results( CONFUSION MATRIX) ** 
	confusionMat, err := evaluation.GetConfusionMatrix(test, predictions)
	fmt.Println(evaluation.GetSummary(confusionMat))
}