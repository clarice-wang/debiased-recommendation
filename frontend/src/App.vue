<template>
  <div id="app">
<!--    <img src="./assets/logo.png">-->
    <survey :survey="survey"></survey>
  </div>
</template>

<script>
  import * as SurveyVue from "survey-vue";
  import "bootstrap/dist/css/bootstrap.css";
  var Survey = SurveyVue.Survey;
  Survey.cssType = "bootstrap";

  import "inputmask/dist/inputmask/phone-codes/phone.js";

  export default {
    name: "app",
    components: {
      'survey' : Survey,
    },
    data: function() {
      var json = {
        title: "Survey Example",
        showProgressBar: "top",
        pages: [
          {
            questions: [
              {
                type: "radiogroup",
                name: "grade",
                title: "What is your class standing?",
                isRequired: true,
                choices: [
                        "Freshman",
                        "Sophomore",
                        "Junior",
                        "Senior",
                        "Graduate Student",
                        "Prefer not to disclose"
                ]
              },
              {
                type: "radiogroup",
                name: "gender",
                title: "What gender do you identify as?",
                isRequired: true,
                choices: [
                  "Male",
                  "Female",
                  "Other"
                ]
              },
              {
                type: "dropdown",
                name: "career",
                title: "What is your current major/minor/concentration?",
                isRequired: true,
                hasOther: true,
                choices: [
                  "Agriculture, Food, and Natural Resources",
                  "Architecture and Construction",
                  "Arts, Audio/Video Technology, and Communications",
                  "Business, Management, and Administration",
                  "Education and Training",
                  "Finance",
                  "Government and Public Administration",
                  "Health Science",
                  "Hospitality and Tourism",
                  "Human Services",
                  "Information Technology",
                  "Law, Public Safety, Corrections, and Security",
                  "Manufacturing",
                  "Marketing",
                  "Science, Technology, Engineering, and Mathematics",
                  "Transportation, Distribution, and Logistics"
                ]
              },
              {
                type: "radiogroup",
                name: "set",
                title: "How “set” are you on your choice of major/minor/concentration?",
                isRequired: true,
                choices: [
                  "Already determined/Unlikely to change",
                  "Still open to suggestions"
                ]
              },
              {
                type: "radiogroup",
                name: "disparity",
                title: "Do you perceive a gender disparity in your selected major/minor/concentration?",
                isRequired: true,
                choices: [
                  "Female-dominated",
                  "Male-dominated",
                  "Neither",
                  "I don't know"
                ]
              },
              {
                type: "panel",
                name: "femaleormaledominated",
                title: "If you are a female, please answer (a). If you are a male, please answer (b). Otherwise,\n" +
                    "you can choose to answer either (a) or (b) but not both.",
                elements: [
                    {
                        type: "radiogroup",
                        name: "femaledominated",
                        title: "(a) Please indicate whether you agree with the following statement or not: “If I am a\n" +
                            "female, I do not want to choose a career that is male-dominated”",
                        choices: [
                            "Strongly Agree",
                            "Agree",
                            "Neither agree nor disagree",
                            "Disagree",
                            "Strongly disagree"
                        ]
                    },
                    {
                        type: "radiogroup",
                        name: "maledominated ",
                        title: "(b) Please indicate whether you agree with the following statement or not: “If I am a\n" +
                            "male, I do not want to choose a career that is female-dominated”",
                        choices: [
                            "Strongly Agree",
                            "Agree",
                            "Neither agree nor disagree",
                            "Disagree",
                            "Strongly disagree"
                        ]
                    }
                ]
              },
              {
                type: "radiogroup",
                name: "opinion",
                title: "Please indicate whether you agree with the following statement or not: “A gender\n" +
                        "stereotype in career selection is undesirable since it limits women's and men's\n" +
                        "capacity to develop their personal abilities and pursue their professional careers”",
                isRequired: true,
                choices: [
                  "Strongly Agree",
                  "Agree",
                  "Neither agree nor disagree",
                  "Disagree",
                  "Strongly disagree"
                ]
              }
            ]
          },
          {
            questions: [
              {
                type: "matrix",
                name: "yesorno",
                title: "Please tell us whether you like the following items or not.",
                isRequired: true,
                columns: [
                  {
                    value: 1,
                    text: "Yes"
                  },
                  {
                    value: 2,
                    text: "No"
                  },
                  {
                    value: 3,
                    text: "I don't know"
                  }
                ],
                rows:[
                  {
                    value: "anime",
                    text: "Anime"
                  },
                  {
                    value: "arianagrande",
                    text: "Ariana Grande"
                  },
                  {
                    value: "australia",
                    text: "Australia"
                  },
                  {
                    value: "avicii",
                    text: "Avicii"
                  },
                  {
                    value: "barackobama",
                    text: "Barack Obama"
                  },
                  {
                    value: "candycrushsaga",
                    text: "Candy Crush Saga"
                  },
                  {
                    value: "chanel",
                    text: "Chanel"
                  },
                  {
                    value: "coldplay",
                    text: "Coldplay"
                  },
                  {
                    value: "converse",
                    text: "Converse"
                  },
                  {
                    value: "eminem",
                    text: "Eminem"
                  },
                  {
                    value: "enriqueiglesias",
                    text: "Enrique Iglesias"
                  },
                  {
                    value: "gopro",
                    text: "GoPro"
                  },
                  {
                    value: "harrypotter",
                    text: "Harry Potter"
                  },
                  {
                    value: "howimetyourmother",
                    text: "How I Met Your Mother"
                  },
                  {
                    value: "hugging",
                    text: "Hugging"
                  },
                  {
                    value: "hughjackman",
                    text: "Hugh Jackman"
                  },
                  {
                    value: "minecraft",
                    text: "Minecraft"
                  },
                  {
                    value: "music",
                    text: "Music"
                  },
                  {
                    value: "nationalgeographic",
                    text: "National Geographic"
                  },
                  {
                    value: "nutella",
                    text: "Nutella"
                  },
                  {
                    value: "ofmiceandmen",
                    text: "Of Mice & Men"
                  },
                  {
                    value: "phineasandferb",
                    text: "Phineas and Ferb"
                  },
                  {
                    value: "positiveoutlooks",
                    text: "Positive Outlooks"
                  },
                  {
                    value: "rihanna",
                    text: "Rihanna"
                  },
                  {
                    value: "roalddahl",
                    text: "Roald Dahl"
                  },
                  {
                    value: "running",
                    text: "Running"
                  },
                  {
                    value: "skittles",
                    text: "Skittles"
                  },
                  {
                    value: "target",
                    text: "Target"
                  },
                  {
                    value: "ted",
                    text: "TED"
                  },
                  {
                    value: "thebeatles",
                    text: "The Beatles"
                  },
                  {
                    value: "thebigbangtheory",
                    text: "The Big Bang Theory"
                  },
                  {
                    value: "theolympicgames",
                    text: "The Olympic Games"
                  },
                  {
                    value: "thexfactor",
                    text: "The X Factor"
                  },
                  {
                    value: "youtube",
                    text: "Youtube"
                  }
                ]
              }
            ]
          },
         {
           questions: [
             {
               type: "radiogroup",
               name: "liketouse",
               title: "Please indicate your agreement with the following statement: “I would like to use a career\n"+
                       "recommendation system like this”",
               isRequired: true,
               choices: [
                 "Strongly Agree",
                 "Agree",
                 "Neither agree nor disagree",
                 "Disagree",
                 "Strongly disagree"
               ]
             },
             {
               type: "radiogroup",
               name: "recommend",
               title: "Please indicate your agreement with the following statement: “I would like to\n"+
               "recommend the system to my friends”",
               isRequired: true,
               choices: [
                 "Strongly Agree",
                 "Agree",
                 "Neither agree nor disagree",
                 "Disagree",
                 "Strongly disagree"
               ]
             },
             {
               type: "comment",
               name: "feedback",
               title: "Any other comments/feedback you would like to share with us?",
             }
           ]
          }
        ],
        completedHtml: "<p>Your answers are:</p>" +
          "<p>Class Standing: <b>{grade}</b></p>" +
          "<p>Gender: <b>{gender}</b></p>" +
          "<p>Current Career Choice: <b>{career}</b></p>" +
          "<p>How set are you on your above choice: <b>{set}</b></p>" +
          "<p>Gender disparity in your career choice: <b>{disparity}</b></p>" +
          "<p>Preference on Male/Female-Dominated Career Choice: <b>{femaledominated} {maledominated}</b></p>" +
          "<p>Gender Stereotype in career choice is undesirable: <b>{opinion}</b></p>" +
          "<p>Anime: <b>{yesorno.anime}</b></p>" +
          "<p>Ariana Grande: <b>{yesorno.arianagrande}</b></p>" +
          "<p>Australia: <b>{yesorno.australia}</b></p>" +
          "<p>Avicii: <b>{yesorno.avicii}</b></p>" +
          "<p>Barack Obama: <b>{yesorno.barackobama}</b></p>" +
          "<p>Candy Crush Saga: <b>{yesorno.candycrushsaga}</b></p>" +
          "<p>Chanel: <b>{yesorno.chanel}</b></p>" +
          "<p>Coldplay: <b>{yesorno.coldplay}</b></p>" +
          "<p>Converse: <b>{yesorno.converse}</b></p>" +
          "<p>Eminem: <b>{yesorno.eminem}</b></p>" +
          "<p>I would like to use a career recommendation system like this: <b>{liketouse}</b></p>" +
          "<p>I would like to use recommend the system to my friends: <b>{recommend}</b></p>" +
          "<p>Comments/Feedback: <b>{feedback}</b></p>"
      };
      var model = new SurveyVue.Model(json);
      return {
        survey: model
      };
    }
  };
</script>

<style>
#app {
  font-family: "Avenir", Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  color: #2c3e50;
}
</style>