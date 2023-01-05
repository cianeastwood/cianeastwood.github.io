#!/usr/bin/env bash

# Copy all files from repo to informatics personal webpage directory
rsync -r --exclude '.git' /afs/inf.ed.ac.uk/user/s16/s1668298/cianeastwood.github.io/ /public/homepages/s1668298/web

# Change permissons so files can be served
find /public/homepages/s1668298/web -type d -exec chmod 755 {} \;
find /public/homepages/s1668298/web -type f -exec chmod 644 {} \; 
