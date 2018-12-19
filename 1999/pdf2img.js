#!/usr/bin/env node
// apt i graphicsmagick
var fs      = require('fs');
var path    = require('path');
var pdf2img = require('pdf2img');

var input   = arg=process.argv[2] // __dirname + '/test.pdf';

pdf2img.setOptions({
  type: 'png',                                // png or jpg, default jpg
  size: 1024,                                 // default 1024
  density: 600,                               // default 600
  // outputdir: __dirname + path.sep + 'output', // output folder
  // outputname: 'out',                         // output file name
  // page: null                                  // convert selected page
});

pdf2img.convert(input, function(err, info) {
  if (err) console.log(err)
  else console.log(info);
});