//
// Created on 2019/2/27.
//
#include "lexical.h"
#include "parser.h"

#include <iostream>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <string>
#include <fstream>
#include <vector>

using namespace std;

std::ifstream inputFile;
std::string lexeme;
char nextChar;
CHARCLASS charClass;
TOKENS nextToken;
std::vector<TOKENS> tokens;
std::vector<std::string> lexemes;
int indentCount;
bool endProgram;
bool string_literal;
bool indent;
int numOfIndents;
int lineNumber;
string outputStream;
void getChar();
void lex();
void processSpaces();
void lookup(char);
void addChar();

void parse();

void log();
ofstream logFile;


string tksnames[] = {
    "IDENTIFIER",
    "INTEGER",
    "STRING_LITERAL",
    "ADD_OP",
    "SUB_OP",
    "MULT_OP",
    "DIV_OP",
    "ASSIGN_OP",
    "COND_EQUAL",
    "COND_GTEQ",
    "COND_GT",
    "COND_LTEQ",
    "COND_LT",
    "COND_NOT_OP",
    "COND_AND",
    "COND_OR",
    "LEFT_PAREN",
    "RIGHT_PAREN",
    "STRING_QUOTE",
    "CHAR_QUOTE",
    "COLON",
    "COMMA",
    "INDENT",
    "LINEBREAK",
    "END"
};


int run(const char* filename){
    lineNumber = 1;
    indentCount = 4;
    numOfIndents = 0;
    endProgram = false;
    string_literal = false;
    indent = true;
    outputStream = "";
    logFile.open("log_mypython.log");
    try{
            inputFile.open(filename);

    }catch(string s){
        printf("Input: mypython <file.py>\n");
    }catch(...){
        printf("Input: mypython <file.py>\n");
    }


    if(inputFile.is_open()){
        //读取代码内容
        getChar();
        lex();
        do{
            log("Token in MAIN file loop BEFORE parsing is '%s'\n", lexeme.c_str());
            parse();
            log("Token in MAIN file loop AFTER parsing is '%s'\n", lexeme.c_str());
            if(endProgram) break;
        }while(nextToken != END);
    }

    inputFile.close();

    //打印数据时的换行和缩进
    log("PRINTING LEXEMES\n");
    for(int i = 0; i < lexemes.size(); i++){
        //向字符串中打印数据
        if(lexemes[i] == "\t") log("indent ");
        else if(lexemes[i] == "\n") log("linebreak ");
        else log("%s ", lexemes[i].c_str());
    }
    log("\n");

    log("PRINTING TOKENS\n");
    for(int i = 0; i < tokens.size(); i++){
        log("%s ", tksnames[tokens[i]].c_str());
        //cout << tksnames[tokens[i]] << " ";
    }
    log("\n");

    //printf("\nPYTHON PROGRAM OUTPUT\n%s\n", outputStream.c_str());

    //outputstream包含由python文件生成的所有输出
    cout << outputStream << endl;

    logFile.close();

    return 0;
}

void log(const char* line, ...){
    va_list argptr;
    va_start(argptr, line);
    char buffer[1024];
    vsnprintf(buffer, 256, line, argptr);
    logFile << buffer;
    //清空数组和指针
    memset(buffer, 0, sizeof(buffer));
    va_end(argptr);
}
