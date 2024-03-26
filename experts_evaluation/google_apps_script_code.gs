function createFormFromJSON() {
  // Replace 'your-json-file-id' with the actual ID of your JSON file
  // Click share in the json file
  // https://drive.google.com/file/d/your-file-id/view

  var jsonFileId = 'your_file_id';


  var jsonFile = DriveApp.getFileById(jsonFileId);
  var jsonString = jsonFile.getBlob().getDataAsString();
  var jsonData = JSON.parse(jsonString);
c:\Users\User\OneDrive - aueb.gr\Documents\Repos\SemEval\Human_evaluation\cot_inspection\forms\sample.json
  // Extract form title and description
  var formTitle = jsonData.formTitle || 'Untitled Form';
  var formDescription = jsonData.formDescription || '';

  // Create a new form
  var form = FormApp.create(formTitle);

  // Set form description
  form.setDescription(formDescription);

  // First section
  var first = jsonData.first_section || [];

  for (var j = 0; j < first.length; j++) {
      var fieldData = first[j];
      var fieldType = fieldData.type;
      var fieldTitle = fieldData.title;
      var isRequired = fieldData.required || false;

      switch (fieldType.toLowerCase()) {
        case 'text':
          form.addTextItem().setTitle(fieldTitle).setRequired(isRequired);
          break;
        case 'description':
          var descriptionField = form.addSectionHeaderItem();
          descriptionField.setTitle(fieldTitle);
          descriptionField.setHelpText(fieldData.details || '');
          break;
        case 'multiplechoice':
          var multipleChoiceField = form.addMultipleChoiceItem();
          var choices = fieldData.choices || [];
          // multipleChoiceField.setTitle(fieldTitle).setChoiceValues(['True', 'False']);
          multipleChoiceField.setTitle(fieldTitle).setChoiceValues(choices);

          break;
        // Add more cases for other field types as needed

        default:
          Logger.log('Unsupported field type: ' + fieldType);
      

      }
  }


  // Loop through sections in the JSON file and add corresponding form items
  var sections = jsonData.sections || [];
  for (var i = 0; i < sections.length; i++) {
    var sectionData = sections[i];
    var sectionTitle = sectionData.title;
    var sectionDescription = sectionData.description;
    var sectionFields = sectionData.fields || [];
    
    // Add a page break to create a new section
    var sectionHeader = form.addPageBreakItem();

    sectionHeader.setTitle(sectionTitle);
    sectionHeader.setHelpText(sectionDescription);
    // Add a section header for each section


    // Loop through fields in the section and add corresponding form items
    for (var j = 0; j < sectionFields.length; j++) {
      var fieldData = sectionFields[j];
      var fieldType = fieldData.type;
      var fieldTitle = fieldData.title;
      var isRequired = fieldData.required || false;


      switch (fieldType.toLowerCase()) {
        case 'text':
          form.addTextItem().setTitle(fieldTitle).setRequired(isRequired);
          break;
        case 'description':
          var descriptionField = form.addSectionHeaderItem();
          descriptionField.setTitle(fieldTitle);
          descriptionField.setHelpText(fieldData.details || '');
          break;
        case 'multiplechoice':
          var multipleChoiceField = form.addMultipleChoiceItem();
          var choices = fieldData.choices || [];
          // multipleChoiceField.setTitle(fieldTitle).setChoiceValues(['True', 'False']);
          multipleChoiceField.setTitle(fieldTitle).setChoiceValues(choices);
          multipleChoiceField.setHelpText(fieldData.details || '');
          break;
        // Add more cases for other field types as needed

        default:
          Logger.log('Unsupported field type: ' + fieldType);
      

      }
    }
  }
  // Log form URLs
  Logger.log('Published URL: ' + form.getPublishedUrl());
  Logger.log('Editor URL: ' + form.getEditUrl());
}