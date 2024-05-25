//Progress Bar
$(document).ready(function () { 
    var currentGfgStep, nextGfgStep, previousGfgStep; 
    var opacity; 
    var current = 1; 
    var steps = $("fieldset").length; 
  
    setProgressBar(current); 
  

/*
    $(".next-step").click(function () { 
  
        currentGfgStep = $(this).parent(); 
        nextGfgStep = $(this).parent().next(); 
  
        $("#progressbar li").eq($("fieldset") 
            .index(nextGfgStep)).addClass("active"); 
  
        nextGfgStep.show(); 
        currentGfgStep.animate({ opacity: 0 }, { 
            step: function (now) { 
                opacity = 1 - now; 
  
                currentGfgStep.css({ 
                    'display': 'none', 
                    'position': 'relative'
                }); 
                nextGfgStep.css({ 'opacity': opacity }); 
            }, 
            duration: 500 
        }); 
        setProgressBar(++current); 
    });  */
    
    $('#next-step1').click(function () { 
        currentGfgStep = $('#next-step1').parent(); 
        nextGfgStep = $('#next-step1').parent().next(); 

        $("#progressbar li").eq($("fieldset") 
            .index(nextGfgStep)).addClass("active"); 

        nextGfgStep.show(); 
        currentGfgStep.animate({ opacity: 0 }, { 
            step: function (now) { 
                opacity = 1 - now; 
                currentGfgStep.css({ 
                    'display': 'none', 
                    'position': 'relative'
                }); 
                nextGfgStep.css({ 'opacity': opacity }); 
            }, 
            duration: 500 
        }); 
        setProgressBar(++current);
        document.getElementById("fieldset2").hidden = false;
        //showCBlist(grid.getData());
        showradiolist(grid.getData());
        });   
    


    $(".previous-step").click(function () { 
        
        currentGfgStep = $(this).parent(); 
        previousGfgStep = $(this).parent().prev(); 
  
        $("#progressbar li").eq($("fieldset") 
            .index(currentGfgStep)).removeClass("active"); 
  
        previousGfgStep.show(); 
  
        currentGfgStep.animate({ opacity: 0 }, { 
            step: function (now) { 
                opacity = 1 - now; 
  
                currentGfgStep.css({ 
                    'position': 'relative',
                    'hidden': 'true'
                }); 
                previousGfgStep.css({ 'opacity': opacity }); 
            }, 
            duration: 500 
        }); 
        setProgressBar(--current);
        next_step2.style.display = "none"; //Rende invisibile il bottone 
    }); 
  
    function setProgressBar(currentStep) { 
        var percent = parseFloat(100 / steps) * current; 
        percent = percent.toFixed(); 
        $(".progress-bar") 
            .css("width", percent + "%") 
    } 
  
    $(".submit").click(function () { 
        return false; 
    }) 
}); 

function showProgress(){
        var progress = document.getElementById('progressbar');
        progress.style.display = "block";
    }

    //Check if the file is .csv 
    var file = document.getElementById('csv-file');

    file.onchange = function(e) {
    var ext = this.value.match(/\.([^\.]+)$/)[1];
    switch (ext) {
        case 'csv':

            break;
        default:
            alert('Insert a csv file');
            this.value = '';
        }
    };


    var csvFile = document.getElementById("csv-file");
    var visual_grid = document.getElementById("grid_csv");
    var next_step1 = document.getElementById("next-step1");
    var next_step2 = document.getElementById("next-step2");
    var csv_data = null;
    var csv_head = null;
    var flag = true;
    var FLAG_ALREADY_POP = false;

    //Handler for deletion of columns
    var deleteColsHandler = function(gridEvent){ 
        //Delete checkboxes and radio of the delated columns
        let incremental = 1;
        new_head = Object.keys(grid.getData()[0]);
        for(i = 0; i<new_head.length; i++){
            if(i+incremental != parseInt(new_head[i])){
                header_old.splice(i, 1);
                incremental++;
            }
        }    
        showCBlist(grid.getData());
        showradiolist(grid.getData());

    };

    //Handler for checking if the old header is modified
    var setCellHandler = function(gridEvent){
        if(gridEvent.rowIds[0] == 1){
            idlabel=gridEvent.colIds[0]-1;
            document.getElementById("cblabel"+idlabel).innerHTML=gridEvent.values[0]
            document.getElementById("rlabel"+idlabel).innerHTML=gridEvent.values[0]
        }
    };



    var csv_data_edit = null;
    var header_old = null;


    //Listener for file input
    csvFile.onchange = function(e){
        
        //This will be done only once the csv is initialized
        
        next_step1.style.display ='block';
        flag = false;
        document.getElementById("fieldset2").hidden = false;
        
        var file = e.target.files[0];

            Papa.parse(file, {
            header: true,
            dynamicTyping: true,
            complete: function(results) {
                csv_data_edit = results.data;
                header_old = Object.keys(csv_data_edit[0]);
                var new_row = {};
                for(i = 0; i<header_old.length; i++){
                    if (header_old[i] !== i) {
                        for(j = 0; j<csv_data_edit.length; j++){
                            //Try catch for checking the existence of a empty row
                            try{
                                Object.defineProperty(csv_data_edit[j], i+1,
                                    Object.getOwnPropertyDescriptor(csv_data_edit[j], header_old[i]));
                                delete csv_data_edit[j][header_old[i]];
                            }catch (error){
                                csv_data_edit.pop();
                            }   
                            
                        }
                    }
                    new_row[i+1] = header_old[i]; 
                }
                csv_data_edit.unshift(new_row);
                
                // overwrite grid with new data
                grid = new DataGridXL("grid", {
                    data: results.data,
                    allowEditCells: true,
                });
                console.log(grid.getData());
                //Activation of delete listener
                grid.events.on('deletecols', deleteColsHandler);
                grid.events.on('setcellvalues', setCellHandler);
                document.getElementById("fieldset2").hidden = true;
            }
        });
    }
    
    //Sending table data to backend
    var url = document.getElementById("progressbar").getAttribute('data-url');
    function getIndepVars(){
        var indep_var = [];
        for(i = 0; i<header_old.length; i++){
            var checkbox = document.getElementById('cb'+i);
            if(checkbox.checked){
                indep_var.push(checkbox.value);indep_var
            }
        }

        return indep_var;
    }

    function getDepVars(){
        var dep_var = null;
        for(i = 0; i<header_old.length; i++){
            var radio_v = document.getElementById('r'+i);
            if(radio_v.checked){
                dep_var = radio_v.value;
                break;
            }
        }
        return dep_var;
    }

    function sendData(){
        csv_data = grid.getData();
        var ind_var = getIndepVars();
        var dep_var = getDepVars();
        $.ajax({ 
            type: "POST",
            url: url+'senddata',
            data: JSON.stringify({"table": csv_data, "indep_var": ind_var, "dep_var": dep_var}),
            success: function(data){        
                console.log(data);
            },
            error: function(err){
                console.log(err);
            }
            
            
        });
    }

    function redo(){
        grid.redo();
        //Delete checkboxes and radio of the delated columns
        let incremental = 1;
        new_head = Object.keys(grid.getData()[0]);
        for(i = 0; i<new_head.length; i++){
            if(i+incremental != parseInt(new_head[i])){
                header_old.splice(i, 1);
                incremental++;
            }
        }  
        showCBlist(grid.getData());
        showradiolist(grid.getData());
    }

    function undo(){
        grid.undo();
        //Repopulate the cblist and radio list back with the deleted columns
        first_row_value = grid.getData()[0];
        first_row_keys = Object.keys(grid.getData()[0]);
        for(i = 0; i<first_row_keys.length; i++){
            if(i+1 == first_row_keys[i] && header_old[i] != first_row_value[i+1]){
                if(header_old[i] == first_row_value[i+2]){
                    header_old.splice(i, 0, first_row_value[i+1]);
                    i++;
                }
            }
        }
        //showCBlist(grid.getData());
        showradiolist(grid.getData());
    }
    //Function showing the checkboxes
    /*
    function showCBlist(data){
        csv_head = Object.keys(data[0]);
        for(var i = 0; i < csv_head.length; i++) {
            if(FLAG_ALREADY_POP){
                document.getElementById("cblist").innerHTML = '';
                FLAG_ALREADY_POP = false;
            }
            document.getElementById("cblist").innerHTML += '<div class="form-check" id="group_cb_'+i+'"required>\
                                                                <input id="cb'+i+'" class="form-check-input" type="checkbox" value="'+header_old[i]+'" onchange="setVal(this.id, this.type)">\
                                                                    <label class="form-check-label text-left" for="flexCheckDefault"  style="margin-right:100%" id="cblabel'+i+'">\
                                                                    '+header_old[i]+'\
                                                                    </label>\
                                                            </div>';
            if(i == csv_head.length-1){
                FLAG_ALREADY_POP = true;
            }
        }
    }*/

    //Checkboxes and Radio Listener
    function setVal(identifier, type){
        /*
        //Toggle the independent/dependent variable if the relative dependent/independent variable is selected
        var idnumber = identifier.match(/\d+/)[0];
        
        if (type== "radio"){
            document.getElementById("cb"+idnumber).checked = false;
            
        }
        else{
            document.getElementById("r"+idnumber).checked = false;
        }
        //Check if at least one cb is selected and if the radio is selected
        var checkboxes = document.querySelectorAll('[id^=cb]');
        for (let i = 0; i < checkboxes.length; i++) {
            if ($("input[type=radio]:checked").length > 0) {
                if (checkboxes[i].checked){
                    next_step2.style.display = "block";
                    break;
                }
                if(i == checkboxes.length - 1){
                    next_step2.style.display = "none";
                }
            }
        
        }
        */
        if ($("input[type=radio]:checked").length > 0) {
            
                next_step2.style.display = "block";
                
            }
        if ($("input[type=radio]:checked").length == 0){
                next_step2.style.display = "none";
            }
                
    }

    //Function showing the radios
    function showradiolist(data){
        csv_head = Object.keys(data[0]);
        for(var i = 0; i < csv_head.length; i++) {
            if(FLAG_ALREADY_POP){
                document.getElementById("rlist").innerHTML = '';
                FLAG_ALREADY_POP = false;
            }
            document.getElementById("rlist").innerHTML += '<div class="form-check" id="group_r_'+i+'" required>\
                                                            <input class="form-check-input" type="radio" name="flexRadioDefault" id="r'+i+'" onchange="setVal(this.id, this.type)" value='+header_old[i]+'>\
                                                                <label class="form-check-label text-left" for="flexRadioDefault1" id="rlabel'+i+'" style="margin-right:100%;">\
                                                                '+header_old[i]+'\
                                                                </label>\
                                                            </div>';
            if(i == csv_head.length-1){
                FLAG_ALREADY_POP = true;
            }
        }
    }
    
    //SweetAlert2 Firing modal
    const swalWithBootstrapButtons = Swal.mixin({
        customClass: {
          confirmButton: "btn btn-success",
          cancelButton: "btn btn-danger"
        },
        buttonsStyling: false
      });

      
    function fireConfirmModal(){
        swalWithBootstrapButtons.fire({
            title: "Are you sure to start analysis?",
          /*   text: "You won't be able to revert this!", */
            icon: "warning",
            showCancelButton: true,
            confirmButtonText: "Yes, start it!",
            cancelButtonText: "No, cancel!",
            reverseButtons: true
          }).then((result) => {
            if (result.isConfirmed) {
                window.location.href = "/Step3";
            } else if (
              /* Read more about handling dismissals below */
              result.dismiss === Swal.DismissReason.cancel
            ) {
              swalWithBootstrapButtons.fire({
                title: "Analysis Cancelled",
                text: "The analysis was cancelled successfully",
                icon: "error"
              });
            }
          });
    }
      