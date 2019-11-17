<template>
    <div id="app">
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
            Survey
        },
        data() {
            var json = {
                title: "Career Recommendation Survey",
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
                                name: "question1",
                                title: "hello?",
                                isRequired: true,
                                choices: [
                                    "hi",
                                    "hey"
                                ]
                            },
                            {
                                type: "radiogroup",
                                name: "disparity",
                                visibleIf: {question1} = "hey",
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
                                visibleIf: {question1} = "hey",
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
                                visibleIf: {question1} = "hey",
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
                                rows: [
                                    {
                                        value: "10109514234",
                                        text: "CHANEL"
                                    },
                                    {
                                        value: "81221197163",
                                        text: "Cristiano Ronaldo"
                                    },
                                    {
                                        value: "7270241753",
                                        text: "YouTube"
                                    },
                                    {
                                        value: "46299886275",
                                        text: "UFC: Ultimate Fighting Championship"
                                    },
                                    {
                                        value: "8103318119",
                                        text: "Target"
                                    },
                                    {
                                        value: "93625750491",
                                        text: "Philadelphia Phillies"
                                    },
                                    {
                                        value: "29534858696",
                                        text: "The Simpsons"
                                    },
                                    {
                                        value: "10092511675",
                                        text: "Rihanna"
                                    },
                                    {
                                        value: "156794164312",
                                        text: "Harry Potter"
                                    },
                                    {
                                        value: "167115176655082",
                                        text: "VICE"
                                    },
                                    {
                                        value: "22934684677",
                                        text: "The Big Bang Theory"
                                    },
                                    {
                                        value: "34703638602",
                                        text: "Tory Kelly"
                                    },
                                    {
                                        value: "50043151918",
                                        text: "GoPro"
                                    },
                                    {
                                        value: "215478321801967",
                                        text: "Phineas and Ferb"
                                    },
                                    {
                                        value: "244944385603396",
                                        text: "Candy Crush Saga"
                                    },
                                    {
                                        value: "19061333123",
                                        text: "Enrique Iglesias"
                                    },
                                    {
                                        value: "111499100431",
                                        text: "National Association for Gun Rights"
                                    },
                                    {
                                        value: "112678439229",
                                        text: "Pretending to Text in Awkward Situations"
                                    },
                                    {
                                        value: "355374000182",
                                        text: "Shrek"
                                    },
                                    {
                                        value: "2413267546",
                                        text: "Music"
                                    },
                                    {
                                        value: "21392801120",
                                        text: "Mitt Romney"
                                    },
                                    {
                                        value: "115576608453665",
                                        text: "Hugging"
                                    },
                                    {
                                        value: "200736335550",
                                        text: "Michael Kors - The Official Page"
                                    },
                                    {
                                        value: "152560524786653",
                                        text: "Positive Outlooks"
                                    },
                                    {
                                        value: "10212595263",
                                        text: "Metallica"
                                    },
                                    {
                                        value: "12384211026",
                                        text: "Ariana Grande"
                                    },
                                    {
                                        value: "6280019085",
                                        text: "Playboy"
                                    },
                                    {
                                        value: "115678694346",
                                        text: "iTunes"
                                    },
                                    {
                                        value: "24609282673",
                                        text: "Family Guy"
                                    },
                                    {
                                        value: "23497828950",
                                        text: "National Geographic"
                                    },
                                    {
                                        value: "78933666629",
                                        text: "Nutella"
                                    },
                                    {
                                        value: "29092950651",
                                        text: "TED"
                                    },
                                    {
                                        value: "42693447371",
                                        text: "New England Patriots"
                                    },
                                    {
                                        value: "6885814958",
                                        text: "Lil Wayne"
                                    },
                                    {
                                        value: "229899403738458",
                                        text: "Call of Duty"
                                    },
                                    {
                                        value: "110288948999058",
                                        text: "The Bible"
                                    },
                                    {
                                        value: "28837043508",
                                        text: "Eminem"
                                    },
                                    {
                                        value: "6092929747",
                                        text: "The Office"
                                    },
                                    {
                                        value: "7807422276",
                                        text: "How I Met Your Mother"
                                    },
                                    {
                                        value: "406433779383",
                                        text: "Dwayne 'The Rock' Johnson"
                                    },
                                    {
                                        value: "7914733474",
                                        text: "Skittles"
                                    },
                                    {
                                        value: "18807449704",
                                        text: "Mashable"
                                    },
                                    {
                                        value: "11071120908",
                                        text: "Australia"
                                    },
                                    {
                                        value: "309446385759638",
                                        text: "Oscar Wilde"
                                    },
                                    {
                                        value: "79294403301",
                                        text: "The X Factor"
                                    },
                                    {
                                        value: "176063032413299",
                                        text: "Leo Messi"
                                    },
                                    {
                                        value: "167633226631991",
                                        text: "Hugh Jackman"
                                    },
                                    {
                                        value: "6815841748",
                                        text: "Barack Obama"
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
                                title: "Please indicate your agreement with the following statement: “I would like to use a career\n" +
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
                                title: "Please indicate your agreement with the following statement: “I would like to\n" +
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
                    "<p>Barack Obama: <b>{yesorno.6815841748}</b></p>" +
                    "<p>CHANEL: <b>{yesorno.10109514234}</b></p>" +
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