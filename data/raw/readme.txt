|-- dump: cook, english and stackoverflow dump
	** queryToImport = all the steps to import all databases

|-- questionExtraction
	-- base_name (cook, english, stack)
   	-- question = question content
   	-- svm = * Feat_Meta_* = asnwers' best view
		 * fold_*_*_all.svm = questions split by fold, without best view
    	   	 -- class = question with best view, after preprocessing. 
           	 -- cook_bkp = backup of questions split by fold, without best view	


|-- qa_results: multi-view result (Hasan's approach)
	* MV_FEAT_MV_results_*.txt = features of the first level learning also used on the second level
   	* MV_results_*.txt? = features of the first level are not using on the second level

|-- experiments_results_qa: results per quality view (and some files for multi-view result, as for at qa_results). For example, history view applied in cook base: cook_multiview_history_results_cook.txt
what are the files? MV_FEAT_MV_results_cook.txt? MV_results_cook.txt?

|-- qa_data: datasets (already processed) in folds, per answer
	* qa_* = for each dataset 


